import abc
import base64
import contextlib
from io import BytesIO, TextIOWrapper
import itertools
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid
import warnings
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib._animation_data import (
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
class FuncAnimation(TimedAnimation):
    """
    `TimedAnimation` subclass that makes an animation by repeatedly calling
    a function *func*.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.

    func : callable
        The function to call at each frame.  The first argument will
        be the next value in *frames*.   Any additional positional
        arguments can be supplied using `functools.partial` or via the *fargs*
        parameter.

        The required signature is::

            def func(frame, *fargs) -> iterable_of_artists

        It is often more convenient to provide the arguments using
        `functools.partial`. In this way it is also possible to pass keyword
        arguments. To pass a function with both positional and keyword
        arguments, set all arguments as keyword arguments, just leaving the
        *frame* argument unset::

            def func(frame, art, *, y=None):
                ...

            ani = FuncAnimation(fig, partial(func, art=ln, y='foo'))

        If ``blit == True``, *func* must return an iterable of all artists
        that were modified or created. This information is used by the blitting
        algorithm to determine which parts of the figure have to be updated.
        The return value is unused if ``blit == False`` and may be omitted in
        that case.

    frames : iterable, int, generator function, or None, optional
        Source of data to pass *func* and each frame of the animation

        - If an iterable, then simply use the values provided.  If the
          iterable has a length, it will override the *save_count* kwarg.

        - If an integer, then equivalent to passing ``range(frames)``

        - If a generator function, then must have the signature::

             def gen_function() -> obj

        - If *None*, then equivalent to passing ``itertools.count``.

        In all of these cases, the values in *frames* is simply passed through
        to the user-supplied *func* and thus can be of any type.

    init_func : callable, optional
        A function used to draw a clear frame. If not given, the results of
        drawing from the first item in the frames sequence will be used. This
        function will be called once before the first frame.

        The required signature is::

            def init_func() -> iterable_of_artists

        If ``blit == True``, *init_func* must return an iterable of artists
        to be re-drawn. This information is used by the blitting algorithm to
        determine which parts of the figure have to be updated.  The return
        value is unused if ``blit == False`` and may be omitted in that case.

    fargs : tuple or None, optional
        Additional arguments to pass to each call to *func*. Note: the use of
        `functools.partial` is preferred over *fargs*. See *func* for details.

    save_count : int, optional
        Fallback for the number of values from *frames* to cache. This is
        only used if the number of frames cannot be inferred from *frames*,
        i.e. when it's an iterator without length or a generator.

    interval : int, default: 200
        Delay between frames in milliseconds.

    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.

    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.

    blit : bool, default: False
        Whether blitting is used to optimize drawing.  Note: when using
        blitting, any animated artists will be drawn according to their zorder;
        however, they will be drawn on top of any previous artists, regardless
        of their zorder.

    cache_frame_data : bool, default: True
        Whether frame data is cached.  Disabling cache might be helpful when
        frames contain large objects.
    """

    def __init__(self, fig, func, frames=None, init_func=None, fargs=None, save_count=None, *, cache_frame_data=True, **kwargs):
        if fargs:
            self._args = fargs
        else:
            self._args = ()
        self._func = func
        self._init_func = init_func
        self._save_count = save_count
        if frames is None:
            self._iter_gen = itertools.count
        elif callable(frames):
            self._iter_gen = frames
        elif np.iterable(frames):
            if kwargs.get('repeat', True):
                self._tee_from = frames

                def iter_frames(frames=frames):
                    this, self._tee_from = itertools.tee(self._tee_from, 2)
                    yield from this
                self._iter_gen = iter_frames
            else:
                self._iter_gen = lambda: iter(frames)
            if hasattr(frames, '__len__'):
                self._save_count = len(frames)
                if save_count is not None:
                    _api.warn_external(f'You passed in an explicit save_count={save_count!r} which is being ignored in favor of len(frames)={len(frames)!r}.')
        else:
            self._iter_gen = lambda: iter(range(frames))
            self._save_count = frames
            if save_count is not None:
                _api.warn_external(f'You passed in an explicit save_count={save_count!r} which is being ignored in favor of frames={frames!r}.')
        if self._save_count is None and cache_frame_data:
            _api.warn_external(f'frames={frames!r} which we can infer the length of, did not pass an explicit *save_count* and passed cache_frame_data={cache_frame_data!r}.  To avoid a possibly unbounded cache, frame data caching has been disabled. To suppress this warning either pass `cache_frame_data=False` or `save_count=MAX_FRAMES`.')
            cache_frame_data = False
        self._cache_frame_data = cache_frame_data
        self._save_seq = []
        super().__init__(fig, **kwargs)
        self._save_seq = []

    def new_frame_seq(self):
        return self._iter_gen()

    def new_saved_frame_seq(self):
        if self._save_seq:
            self._old_saved_seq = list(self._save_seq)
            return iter(self._old_saved_seq)
        elif self._save_count is None:
            frame_seq = self.new_frame_seq()

            def gen():
                try:
                    while True:
                        yield next(frame_seq)
                except StopIteration:
                    pass
            return gen()
        else:
            return itertools.islice(self.new_frame_seq(), self._save_count)

    def _init_draw(self):
        super()._init_draw()
        if self._init_func is None:
            try:
                frame_data = next(self.new_frame_seq())
            except StopIteration:
                warnings.warn('Can not start iterating the frames for the initial draw. This can be caused by passing in a 0 length sequence for *frames*.\n\nIf you passed *frames* as a generator it may be exhausted due to a previous display or save.')
                return
            self._draw_frame(frame_data)
        else:
            self._drawn_artists = self._init_func()
            if self._blit:
                if self._drawn_artists is None:
                    raise RuntimeError('The init_func must return a sequence of Artist objects.')
                for a in self._drawn_artists:
                    a.set_animated(self._blit)
        self._save_seq = []

    def _draw_frame(self, framedata):
        if self._cache_frame_data:
            self._save_seq.append(framedata)
            self._save_seq = self._save_seq[-self._save_count:]
        self._drawn_artists = self._func(framedata, *self._args)
        if self._blit:
            err = RuntimeError('The animation function must return a sequence of Artist objects.')
            try:
                iter(self._drawn_artists)
            except TypeError:
                raise err from None
            for i in self._drawn_artists:
                if not isinstance(i, mpl.artist.Artist):
                    raise err
            self._drawn_artists = sorted(self._drawn_artists, key=lambda x: x.get_zorder())
            for a in self._drawn_artists:
                a.set_animated(self._blit)
    save_count = _api.deprecate_privatize_attribute('3.7')