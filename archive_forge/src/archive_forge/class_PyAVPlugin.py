from fractions import Fraction
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
import av
import av.filter
import numpy as np
from numpy.lib.stride_tricks import as_strided
from ..core import Request
from ..core.request import URI_BYTES, InitializationError, IOMode
from ..core.v3_plugin_api import ImageProperties, PluginV3
class PyAVPlugin(PluginV3):
    """Support for pyAV as backend.

    Parameters
    ----------
    request : iio.Request
        A request object that represents the users intent. It provides a
        standard interface to access various the various ImageResources and
        serves them to the plugin as a file object (or file). Check the docs for
        details.
    container : str
        Only used during `iio_mode="w"`! If not None, overwrite the default container
        format chosen by pyav.
    kwargs : Any
        Additional kwargs are forwarded to PyAV's constructor.

    """

    def __init__(self, request: Request, *, container: str=None, **kwargs) -> None:
        """Initialize a new Plugin Instance.

        See Plugin's docstring for detailed documentation.

        Notes
        -----
        The implementation here stores the request as a local variable that is
        exposed using a @property below. If you inherit from PluginV3, remember
        to call ``super().__init__(request)``.

        """
        super().__init__(request)
        self._container = None
        self._video_stream = None
        self._video_filter = None
        if request.mode.io_mode == IOMode.read:
            self._next_idx = 0
            try:
                if request._uri_type == 5:
                    self._container = av.open(request.raw_uri, **kwargs)
                else:
                    self._container = av.open(request.get_file(), **kwargs)
                self._video_stream = self._container.streams.video[0]
                self._decoder = self._container.decode(video=0)
            except av.AVError:
                if isinstance(request.raw_uri, bytes):
                    msg = 'PyAV does not support these `<bytes>`'
                else:
                    msg = f'PyAV does not support `{request.raw_uri}`'
                raise InitializationError(msg) from None
        else:
            self.frames_written = 0
            file_handle = self.request.get_file()
            filename = getattr(file_handle, 'name', None)
            extension = self.request.extension or self.request.format_hint
            if extension is None:
                raise InitializationError("Can't determine output container to use.")
            try:
                setattr(file_handle, 'name', filename or 'tmp' + extension)
            except AttributeError:
                pass
            try:
                self._container = av.open(file_handle, mode='w', format=container, **kwargs)
            except ValueError:
                raise InitializationError(f'PyAV can not write to `{self.request.raw_uri}`')

    def read(self, *, index: int=..., format: str='rgb24', filter_sequence: List[Tuple[str, Union[str, dict]]]=None, filter_graph: Tuple[dict, List]=None, constant_framerate: bool=None, thread_count: int=0, thread_type: str=None) -> np.ndarray:
        """Read frames from the video.

        If ``index`` is an integer, this function reads the index-th frame from
        the file. If ``index`` is ... (Ellipsis), this function reads all frames
        from the video, stacks them along the first dimension, and returns a
        batch of frames.

        Parameters
        ----------
        index : int
            The index of the frame to read, e.g. ``index=5`` reads the 5th
            frame. If ``...``, read all the frames in the video and stack them
            along a new, prepended, batch dimension.
        format : str
            Set the returned colorspace. If not None (default: rgb24), convert
            the data into the given format before returning it. If ``None``
            return the data in the encoded format if it can be expressed as a
            strided array; otherwise raise an Exception.
        filter_sequence : List[str, str, dict]
            If not None, apply the given sequence of FFmpeg filters to each
            ndimage. Check the (module-level) plugin docs for details and
            examples.
        filter_graph : (dict, List)
            If not None, apply the given graph of FFmpeg filters to each
            ndimage. The graph is given as a tuple of two dicts. The first dict
            contains a (named) set of nodes, and the second dict contains a set
            of edges between nodes of the previous dict. Check the (module-level)
            plugin docs for details and examples.
        constant_framerate : bool
            If True assume the video's framerate is constant. This allows for
            faster seeking inside the file. If False, the video is reset before
            each read and searched from the beginning. If None (default), this
            value will be read from the container format.
        thread_count : int
            How many threads to use when decoding a frame. The default is 0,
            which will set the number using ffmpeg's default, which is based on
            the codec, number of available cores, threadding model, and other
            considerations.
        thread_type : str
            The threading model to be used. One of

            - `"SLICE"`: threads assemble parts of the current frame
            - `"FRAME"`: threads may assemble future frames
            - None (default): Uses ``"FRAME"`` if ``index=...`` and ffmpeg's
              default otherwise.


        Returns
        -------
        frame : np.ndarray
            A numpy array containing loaded frame data.

        Notes
        -----
        Accessing random frames repeatedly is costly (O(k), where k is the
        average distance between two keyframes). You should do so only sparingly
        if possible. In some cases, it can be faster to bulk-read the video (if
        it fits into memory) and to then access the returned ndarray randomly.

        The current implementation may cause problems for b-frames, i.e.,
        bidirectionaly predicted pictures. I lack test videos to write unit
        tests for this case.

        Reading from an index other than ``...``, i.e. reading a single frame,
        currently doesn't support filters that introduce delays.

        """
        if index is ...:
            props = self.properties(format=format)
            uses_filter = self._video_filter is not None or filter_graph is not None or filter_sequence is not None
            self._container.seek(0)
            if not uses_filter and props.shape[0] != 0:
                frames = np.empty(props.shape, dtype=props.dtype)
                for idx, frame in enumerate(self.iter(format=format, filter_sequence=filter_sequence, filter_graph=filter_graph, thread_count=thread_count, thread_type=thread_type or 'FRAME')):
                    frames[idx] = frame
            else:
                frames = np.stack([x for x in self.iter(format=format, filter_sequence=filter_sequence, filter_graph=filter_graph, thread_count=thread_count, thread_type=thread_type or 'FRAME')])
            self._video_stream.close()
            self._video_stream = self._container.streams.video[0]
            return frames
        if thread_type is not None and thread_type != self._video_stream.thread_type:
            self._video_stream.thread_type = thread_type
        if thread_count != 0 and thread_count != self._video_stream.codec_context.thread_count:
            self._video_stream.codec_context.thread_count = thread_count
        if constant_framerate is None:
            constant_framerate = not self._container.format.variable_fps
        self._seek(index, constant_framerate=constant_framerate)
        desired_frame = next(self._decoder)
        self._next_idx += 1
        self.set_video_filter(filter_sequence, filter_graph)
        if self._video_filter is not None:
            desired_frame = self._video_filter.send(desired_frame)
        return self._unpack_frame(desired_frame, format=format)

    def iter(self, *, format: str='rgb24', filter_sequence: List[Tuple[str, Union[str, dict]]]=None, filter_graph: Tuple[dict, List]=None, thread_count: int=0, thread_type: str=None) -> np.ndarray:
        """Yield frames from the video.

        Parameters
        ----------
        frame : np.ndarray
            A numpy array containing loaded frame data.
        format : str
            Convert the data into the given format before returning it. If None,
            return the data in the encoded format if it can be expressed as a
            strided array; otherwise raise an Exception.
        filter_sequence : List[str, str, dict]
            Set the returned colorspace. If not None (default: rgb24), convert
            the data into the given format before returning it. If ``None``
            return the data in the encoded format if it can be expressed as a
            strided array; otherwise raise an Exception.
        filter_graph : (dict, List)
            If not None, apply the given graph of FFmpeg filters to each
            ndimage. The graph is given as a tuple of two dicts. The first dict
            contains a (named) set of nodes, and the second dict contains a set
            of edges between nodes of the previous dict. Check the (module-level)
            plugin docs for details and examples.
        thread_count : int
            How many threads to use when decoding a frame. The default is 0,
            which will set the number using ffmpeg's default, which is based on
            the codec, number of available cores, threadding model, and other
            considerations.
        thread_type : str
            The threading model to be used. One of

            - `"SLICE"` (default): threads assemble parts of the current frame
            - `"FRAME"`: threads may assemble future frames (faster for bulk reading)


        Yields
        ------
        frame : np.ndarray
            A (decoded) video frame.


        """
        self._video_stream.thread_type = thread_type or 'SLICE'
        self._video_stream.codec_context.thread_count = thread_count
        self.set_video_filter(filter_sequence, filter_graph)
        for frame in self._decoder:
            self._next_idx += 1
            if self._video_filter is not None:
                try:
                    frame = self._video_filter.send(frame)
                except StopIteration:
                    break
            if frame is None:
                continue
            yield self._unpack_frame(frame, format=format)
        if self._video_filter is not None:
            for frame in self._video_filter:
                yield self._unpack_frame(frame, format=format)

    def write(self, ndimage: Union[np.ndarray, List[np.ndarray]], *, codec: str=None, is_batch: bool=True, fps: int=24, in_pixel_format: str='rgb24', out_pixel_format: str=None, filter_sequence: List[Tuple[str, Union[str, dict]]]=None, filter_graph: Tuple[dict, List]=None) -> Optional[bytes]:
        """Save a ndimage as a video.

        Given a batch of frames (stacked along the first axis) or a list of
        frames, encode them and add the result to the ImageResource.

        Parameters
        ----------
        ndimage : ArrayLike, List[ArrayLike]
            The ndimage to encode and write to the ImageResource.
        codec : str
            The codec to use when encoding frames. Only needed on first write
            and ignored on subsequent writes.
        is_batch : bool
            If True (default), the ndimage is a batch of images, otherwise it is
            a single image. This parameter has no effect on lists of ndimages.
        fps : str
            The resulting videos frames per second.
        in_pixel_format : str
            The pixel format of the incoming ndarray. Defaults to "rgb24" and can
            be any stridable pix_fmt supported by FFmpeg.
        out_pixel_format : str
            The pixel format to use while encoding frames. If None (default)
            use the codec's default.
        filter_sequence : List[str, str, dict]
            If not None, apply the given sequence of FFmpeg filters to each
            ndimage. Check the (module-level) plugin docs for details and
            examples.
        filter_graph : (dict, List)
            If not None, apply the given graph of FFmpeg filters to each
            ndimage. The graph is given as a tuple of two dicts. The first dict
            contains a (named) set of nodes, and the second dict contains a set
            of edges between nodes of the previous dict. Check the (module-level)
            plugin docs for details and examples.

        Returns
        -------
        encoded_image : bytes or None
            If the chosen ImageResource is the special target ``"<bytes>"`` then
            write will return a byte string containing the encoded image data.
            Otherwise, it returns None.

        Notes
        -----
        When writing ``<bytes>``, the video is finalized immediately after the
        first write call and calling write multiple times to append frames is
        not possible.

        """
        if isinstance(ndimage, list):
            if any((f.shape != ndimage[0].shape for f in ndimage)):
                raise ValueError('All frames should have the same shape')
        elif not is_batch:
            ndimage = np.asarray(ndimage)[None, ...]
        else:
            ndimage = np.asarray(ndimage)
        if self._video_stream is None:
            self.init_video_stream(codec, fps=fps, pixel_format=out_pixel_format)
        self.set_video_filter(filter_sequence, filter_graph)
        for img in ndimage:
            self.write_frame(img, pixel_format=in_pixel_format)
        if self.request._uri_type == URI_BYTES:
            self._flush_writer()
            self._container.close()
            return self.request.get_file().getvalue()

    def properties(self, index: int=..., *, format: str='rgb24') -> ImageProperties:
        """Standardized ndimage metadata.

        Parameters
        ----------
        index : int
            The index of the ndimage for which to return properties. If ``...``
            (Ellipsis, default), return the properties for the resulting batch
            of frames.
        format : str
            If not None (default: rgb24), convert the data into the given format
            before returning it. If None return the data in the encoded format
            if that can be expressed as a strided array; otherwise raise an
            Exception.

        Returns
        -------
        properties : ImageProperties
            A dataclass filled with standardized image metadata.

        Notes
        -----
        This function is efficient and won't process any pixel data.

        The provided metadata does not include modifications by any filters
        (through ``filter_sequence`` or ``filter_graph``).

        """
        video_width = self._video_stream.codec_context.width
        video_height = self._video_stream.codec_context.height
        pix_format = format or self._video_stream.codec_context.pix_fmt
        frame_template = av.VideoFrame(video_width, video_height, pix_format)
        shape = _get_frame_shape(frame_template)
        if index is ...:
            n_frames = self._video_stream.frames
            shape = (n_frames,) + shape
        return ImageProperties(shape=tuple(shape), dtype=_format_to_dtype(frame_template.format), n_images=shape[0] if index is ... else None, is_batch=index is ...)

    def metadata(self, index: int=..., exclude_applied: bool=True, constant_framerate: bool=None) -> Dict[str, Any]:
        """Format-specific metadata.

        Returns a dictionary filled with metadata that is either stored in the
        container, the video stream, or the frame's side-data.

        Parameters
        ----------
        index : int
            If ... (Ellipsis, default) return global metadata (the metadata
            stored in the container and video stream). If not ..., return the
            side data stored in the frame at the given index.
        exclude_applied : bool
            Currently, this parameter has no effect. It exists for compliance with
            the ImageIO v3 API.
        constant_framerate : bool
            If True assume the video's framerate is constant. This allows for
            faster seeking inside the file. If False, the video is reset before
            each read and searched from the beginning. If None (default), this
            value will be read from the container format.

        Returns
        -------
        metadata : dict
            A dictionary filled with format-specific metadata fields and their
            values.

        """
        metadata = dict()
        if index is ...:
            metadata.update({'video_format': self._video_stream.codec_context.pix_fmt, 'codec': self._video_stream.codec.name, 'long_codec': self._video_stream.codec.long_name, 'profile': self._video_stream.profile, 'fps': float(self._video_stream.guessed_rate)})
            if self._video_stream.duration is not None:
                duration = float(self._video_stream.duration * self._video_stream.time_base)
                metadata.update({'duration': duration})
            metadata.update(self.container_metadata)
            metadata.update(self.video_stream_metadata)
            return metadata
        if constant_framerate is None:
            constant_framerate = not self._container.format.variable_fps
        self._seek(index, constant_framerate=constant_framerate)
        desired_frame = next(self._decoder)
        self._next_idx += 1
        metadata.update({'key_frame': bool(desired_frame.key_frame), 'time': desired_frame.time, 'interlaced_frame': bool(desired_frame.interlaced_frame), 'frame_type': desired_frame.pict_type.name})
        metadata.update({item.type.name: item.to_bytes() for item in desired_frame.side_data})
        return metadata

    def close(self) -> None:
        """Close the Video."""
        is_write = self.request.mode.io_mode == IOMode.write
        if is_write and self._video_stream is not None:
            self._flush_writer()
        if self._container is not None:
            self._container.close()
        self.request.finish()

    def __enter__(self) -> 'PyAVPlugin':
        return super().__enter__()

    def init_video_stream(self, codec: str, *, fps: float=24, pixel_format: str=None, max_keyframe_interval: int=None, force_keyframes: bool=None) -> None:
        """Initialize a new video stream.

        This function adds a new video stream to the ImageResource using the
        selected encoder (codec), framerate, and colorspace.

        Parameters
        ----------
        codec : str
            The codec to use, e.g. ``"x264"`` or ``"vp9"``.
        fps : float
            The desired framerate of the video stream (frames per second).
        pixel_format : str
            The pixel format to use while encoding frames. If None (default) use
            the codec's default.
        max_keyframe_interval : int
            The maximum distance between two intra frames (I-frames). Also known
            as GOP size. If unspecified use the codec's default. Note that not
            every I-frame is a keyframe; see the notes for details.
        force_keyframes : bool
            If True, limit inter frames dependency to frames within the current
            keyframe interval (GOP), i.e., force every I-frame to be a keyframe.
            If unspecified, use the codec's default.

        Notes
        -----
        You can usually leave ``max_keyframe_interval`` and ``force_keyframes``
        at their default values, unless you try to generate seek-optimized video
        or have a similar specialist use-case. In this case, ``force_keyframes``
        controls the ability to seek to _every_ I-frame, and
        ``max_keyframe_interval`` controls how close to a random frame you can
        seek. Low values allow more fine-grained seek at the expense of
        file-size (and thus I/O performance).

        """
        stream = self._container.add_stream(codec, fps)
        stream.time_base = Fraction(1 / fps).limit_denominator(int(2 ** 16 - 1))
        if pixel_format is not None:
            stream.pix_fmt = pixel_format
        if max_keyframe_interval is not None:
            stream.gop_size = max_keyframe_interval
        if force_keyframes is not None:
            stream.closed_gop = force_keyframes
        self._video_stream = stream

    def write_frame(self, frame: np.ndarray, *, pixel_format: str='rgb24') -> None:
        """Add a frame to the video stream.

        This function appends a new frame to the video. It assumes that the
        stream previously has been initialized. I.e., ``init_video_stream`` has
        to be called before calling this function for the write to succeed.

        Parameters
        ----------
        frame : np.ndarray
            The image to be appended/written to the video stream.
        pixel_format : str
            The colorspace (pixel format) of the incoming frame.

        Notes
        -----
        Frames may be held in a buffer, e.g., by the filter pipeline used during
        writing or by FFMPEG to batch them prior to encoding. Make sure to
        ``.close()`` the plugin or to use a context manager to ensure that all
        frames are written to the ImageResource.

        """
        pixel_format = av.VideoFormat(pixel_format)
        img_dtype = _format_to_dtype(pixel_format)
        width = frame.shape[2 if pixel_format.is_planar else 1]
        height = frame.shape[1 if pixel_format.is_planar else 0]
        av_frame = av.VideoFrame(width, height, pixel_format.name)
        if pixel_format.is_planar:
            for idx, plane in enumerate(av_frame.planes):
                plane_array = np.frombuffer(plane, dtype=img_dtype)
                plane_array = as_strided(plane_array, shape=(plane.height, plane.width), strides=(plane.line_size, img_dtype.itemsize))
                plane_array[...] = frame[idx]
        else:
            if pixel_format.name.startswith('bayer_'):
                n_channels = 1
            else:
                n_channels = len(pixel_format.components)
            plane = av_frame.planes[0]
            plane_shape = (plane.height, plane.width)
            plane_strides = (plane.line_size, n_channels * img_dtype.itemsize)
            if n_channels > 1:
                plane_shape += (n_channels,)
                plane_strides += (img_dtype.itemsize,)
            plane_array = as_strided(np.frombuffer(plane, dtype=img_dtype), shape=plane_shape, strides=plane_strides)
            plane_array[...] = frame
        stream = self._video_stream
        av_frame.time_base = stream.codec_context.time_base
        av_frame.pts = self.frames_written
        self.frames_written += 1
        if self._video_filter is not None:
            av_frame = self._video_filter.send(av_frame)
            if av_frame is None:
                return
        if stream.frames == 0:
            stream.width = av_frame.width
            stream.height = av_frame.height
        for packet in stream.encode(av_frame):
            self._container.mux(packet)

    def set_video_filter(self, filter_sequence: List[Tuple[str, Union[str, dict]]]=None, filter_graph: Tuple[dict, List]=None) -> None:
        """Set the filter(s) to use.

        This function creates a new FFMPEG filter graph to use when reading or
        writing video. In the case of reading, frames are passed through the
        filter graph before begin returned and, in case of writing, frames are
        passed through the filter before being written to the video.

        Parameters
        ----------
        filter_sequence : List[str, str, dict]
            If not None, apply the given sequence of FFmpeg filters to each
            ndimage. Check the (module-level) plugin docs for details and
            examples.
        filter_graph : (dict, List)
            If not None, apply the given graph of FFmpeg filters to each
            ndimage. The graph is given as a tuple of two dicts. The first dict
            contains a (named) set of nodes, and the second dict contains a set
            of edges between nodes of the previous dict. Check the
            (module-level) plugin docs for details and examples.

        Notes
        -----
        Changing a filter graph with lag during reading or writing will
        currently cause frames in the filter queue to be lost.

        """
        if filter_sequence is None and filter_graph is None:
            self._video_filter = None
            return
        if filter_sequence is None:
            filter_sequence = list()
        node_descriptors: Dict[str, Tuple[str, Union[str, Dict]]]
        edges: List[Tuple[str, str, int, int]]
        if filter_graph is None:
            node_descriptors, edges = (dict(), [('video_in', 'video_out', 0, 0)])
        else:
            node_descriptors, edges = filter_graph
        graph = av.filter.Graph()
        previous_node = graph.add_buffer(template=self._video_stream)
        for filter_name, argument in filter_sequence:
            if isinstance(argument, str):
                current_node = graph.add(filter_name, argument)
            else:
                current_node = graph.add(filter_name, **argument)
            previous_node.link_to(current_node)
            previous_node = current_node
        nodes = dict()
        nodes['video_in'] = previous_node
        nodes['video_out'] = graph.add('buffersink')
        for name, (filter_name, arguments) in node_descriptors.items():
            if isinstance(arguments, str):
                nodes[name] = graph.add(filter_name, arguments)
            else:
                nodes[name] = graph.add(filter_name, **arguments)
        for from_note, to_node, out_idx, in_idx in edges:
            nodes[from_note].link_to(nodes[to_node], out_idx, in_idx)
        graph.configure()

        def video_filter():
            frame = (yield None)
            while frame is not None:
                graph.push(frame)
                try:
                    frame = (yield graph.pull())
                except av.error.BlockingIOError:
                    frame = (yield None)
                except av.error.EOFError:
                    break
            try:
                graph.push(None)
            except ValueError:
                pass
            while True:
                try:
                    yield graph.pull()
                except av.error.EOFError:
                    break
                except av.error.BlockingIOError:
                    break
        self._video_filter = video_filter()
        self._video_filter.send(None)

    @property
    def container_metadata(self):
        """Container-specific metadata.

        A dictionary containing metadata stored at the container level.

        """
        return self._container.metadata

    @property
    def video_stream_metadata(self):
        """Stream-specific metadata.

        A dictionary containing metadata stored at the stream level.

        """
        return self._video_stream.metadata

    def _unpack_frame(self, frame: av.VideoFrame, *, format: str=None) -> np.ndarray:
        """Convert a av.VideoFrame into a ndarray

        Parameters
        ----------
        frame : av.VideoFrame
            The frame to unpack.
        format : str
            If not None, convert the frame to the given format before unpacking.

        """
        if format is not None:
            frame = frame.reformat(format=format)
        dtype = _format_to_dtype(frame.format)
        shape = _get_frame_shape(frame)
        planes = list()
        for idx in range(len(frame.planes)):
            n_channels = sum([x.bits // (dtype.itemsize * 8) for x in frame.format.components if x.plane == idx])
            av_plane = frame.planes[idx]
            plane_shape = (av_plane.height, av_plane.width)
            plane_strides = (av_plane.line_size, n_channels * dtype.itemsize)
            if n_channels > 1:
                plane_shape += (n_channels,)
                plane_strides += (dtype.itemsize,)
            np_plane = as_strided(np.frombuffer(av_plane, dtype=dtype), shape=plane_shape, strides=plane_strides)
            planes.append(np_plane)
        if len(planes) > 1:
            out = np.concatenate(planes).reshape(shape)
        else:
            out = planes[0]
        return out

    def _seek(self, index, *, constant_framerate: bool=True) -> Generator:
        """Seeks to the frame at the given index."""
        if index == self._next_idx:
            return
        if self._next_idx == 0:
            next(self._decoder)
            self._next_idx += 1
            if index == self._next_idx:
                return
        if not constant_framerate and index > self._next_idx:
            frames_to_yield = index - self._next_idx
        elif not constant_framerate:
            self._container.seek(0)
            self._decoder = self._container.decode(video=0)
            self._next_idx = 0
            frames_to_yield = index
        else:
            sec_delta = 1 / self._video_stream.guessed_rate
            pts_delta = sec_delta / self._video_stream.time_base
            index_pts = int(index * pts_delta)
            self._container.seek(index_pts, stream=self._video_stream)
            self._decoder = self._container.decode(video=0)
            keyframe = next(self._decoder)
            keyframe_time = keyframe.pts * keyframe.time_base
            keyframe_pts = int(keyframe_time / self._video_stream.time_base)
            keyframe_index = keyframe_pts // pts_delta
            self._container.seek(index_pts, stream=self._video_stream)
            self._next_idx = keyframe_index
            frames_to_yield = index - keyframe_index
        for _ in range(frames_to_yield):
            next(self._decoder)
            self._next_idx += 1

    def _flush_writer(self):
        """Flush the filter and encoder

        This will reset the filter to `None` and send EoF to the encoder,
        i.e., after calling, no more frames may be written.

        """
        stream = self._video_stream
        if self._video_filter is not None:
            for av_frame in self._video_filter:
                if stream.frames == 0:
                    stream.width = av_frame.width
                    stream.height = av_frame.height
                for packet in stream.encode(av_frame):
                    self._container.mux(packet)
            self._video_filter = None
        for packet in stream.encode():
            self._container.mux(packet)
        self._video_stream = None