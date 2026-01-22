import itertools
import os
import pickle
import re
import shutil
import string
import tarfile
import time
import zipfile
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import param
from param.parameterized import bothmethod
from .dimension import LabelledData
from .element import Collator, Element
from .ndmapping import NdMapping, UniformNdMapping
from .options import Store
from .overlay import Layout, Overlay
from .util import group_sanitizer, label_sanitizer, unique_iterator
class FileArchive(Archive):
    """
    A file archive stores files on disk, either unpacked in a
    directory or in an archive format (e.g. a zip file).
    """
    exporters = param.List(default=[Pickler], doc='\n        The exporter functions used to convert HoloViews objects into\n        the appropriate format(s).')
    dimension_formatter = param.String('{name}_{range}', doc='\n        A string formatter for the output file based on the\n        supplied HoloViews objects dimension names and values.\n        Valid fields are the {name}, {range} and {unit} of the\n        dimensions.')
    object_formatter = param.Callable(default=simple_name_generator, doc='\n        Callable that given an object returns a string suitable for\n        inclusion in file and directory names. This is what generates\n        the value used in the {obj} field of the filename\n        formatter.')
    filename_formatter = param.String('{dimensions},{obj}', doc='\n        A string formatter for output filename based on the HoloViews\n        object that is being rendered to disk.\n\n        The available fields are the {type}, {group}, {label}, {obj}\n        of the holoviews object added to the archive as well as\n        {timestamp}, {obj} and {SHA}. The {timestamp} is the export\n        timestamp using timestamp_format, {obj} is the object\n        representation as returned by object_formatter and {SHA} is\n        the SHA of the {obj} value used to compress it into a shorter\n        string.')
    timestamp_format = param.String('%Y_%m_%d-%H_%M_%S', doc='\n        The timestamp format that will be substituted for the\n        {timestamp} field in the export name.')
    root = param.String('.', doc='\n        The root directory in which the output directory is\n        located. May be an absolute or relative path.')
    archive_format = param.ObjectSelector(default='zip', objects=['zip', 'tar'], doc="\n        The archive format to use if there are multiple files and pack\n        is set to True. Supported formats include 'zip' and 'tar'.")
    pack = param.Boolean(default=False, doc='\n        Whether or not to pack to contents into the specified archive\n        format. If pack is False, the contents will be output to a\n        directory.\n\n        Note that if there is only a single file in the archive, no\n        packing will occur and no directory is created. Instead, the\n        file is treated as a single-file archive.')
    export_name = param.String(default='{timestamp}', doc='\n        The name assigned to the overall export. If an archive file is\n        used, this is the correspond filename (e.g. of the exporter zip\n        file). Alternatively, if unpack=False, this is the name of the\n        output directory. Lastly, for archives of a single file, this\n        is the basename of the output file.\n\n        The {timestamp} field is available to include the timestamp at\n        the time of export in the chosen timestamp format.')
    unique_name = param.Boolean(default=False, doc='\n       Whether the export name should be made unique with a numeric\n       suffix. If set to False, any existing export of the same name\n       will be removed and replaced.')
    max_filename = param.Integer(default=100, bounds=(0, None), doc='\n       Maximum length to enforce on generated filenames.  100 is the\n       practical maximum for zip and tar file generation, but you may\n       wish to use a lower value to avoid long filenames.')
    flush_archive = param.Boolean(default=True, doc='\n       Flushed the contents of the archive after export.\n       ')
    ffields = {'type', 'group', 'label', 'obj', 'SHA', 'timestamp', 'dimensions'}
    efields = {'timestamp'}

    @classmethod
    def parse_fields(cls, formatter):
        """Returns the format fields otherwise raise exception"""
        if formatter is None:
            return []
        try:
            parse = list(string.Formatter().parse(formatter))
            return {f for f in list(zip(*parse))[1] if f is not None}
        except Exception as e:
            raise SyntaxError(f'Could not parse formatter {formatter!r}') from e

    def __init__(self, **params):
        super().__init__(**params)
        self._files = {}
        self._validate_formatters()

    def _dim_formatter(self, obj):
        if not obj:
            return ''
        key_dims = obj.traverse(lambda x: x.kdims, [UniformNdMapping])
        constant_dims = obj.traverse(lambda x: x.cdims)
        dims = []
        map(dims.extend, key_dims + constant_dims)
        dims = unique_iterator(dims)
        dim_strings = []
        for dim in dims:
            lower, upper = obj.range(dim.name)
            lower, upper = (dim.pprint_value(lower), dim.pprint_value(upper))
            if lower == upper:
                range = dim.pprint_value(lower)
            else:
                range = f'{lower}-{upper}'
            formatters = {'name': dim.name, 'range': range, 'unit': dim.unit}
            dim_strings.append(self.dimension_formatter.format(**formatters))
        return '_'.join(dim_strings)

    def _validate_formatters(self):
        if not self.parse_fields(self.filename_formatter).issubset(self.ffields):
            raise Exception(f'Valid filename fields are: {','.join(sorted(self.ffields))}')
        elif not self.parse_fields(self.export_name).issubset(self.efields):
            raise Exception(f'Valid export fields are: {','.join(sorted(self.efields))}')
        try:
            time.strftime(self.timestamp_format, tuple(time.localtime()))
        except Exception as e:
            raise Exception('Timestamp format invalid') from e

    def add(self, obj=None, filename=None, data=None, info=None, **kwargs):
        """
        If a filename is supplied, it will be used. Otherwise, a
        filename will be generated from the supplied object. Note that
        if the explicit filename uses the {timestamp} field, it will
        be formatted upon export.

        The data to be archived is either supplied explicitly as
        'data' or automatically rendered from the object.
        """
        if info is None:
            info = {}
        if [filename, obj] == [None, None]:
            raise Exception('Either filename or a HoloViews object is needed to create an entry in the archive.')
        elif obj is None and (not self.parse_fields(filename).issubset({'timestamp'})):
            raise Exception('Only the {timestamp} formatter may be used unless an object is supplied.')
        elif [obj, data] == [None, None]:
            raise Exception('Either an object or explicit data must be supplied to create an entry in the archive.')
        elif data and 'mime_type' not in info:
            raise Exception('The mime-type must be supplied in the info dictionary when supplying data directly')
        self._validate_formatters()
        entries = []
        if data is None:
            for exporter in self.exporters:
                rendered = exporter(obj)
                if rendered is None:
                    continue
                data, new_info = rendered
                info = dict(info, **new_info)
                entries.append((data, info))
        else:
            entries.append((data, info))
        for data, info in entries:
            self._add_content(obj, data, info, filename=filename)

    def _add_content(self, obj, data, info, filename=None):
        unique_key, ext = self._compute_filename(obj, info, filename=filename)
        self._files[unique_key, ext] = (data, info)

    def _compute_filename(self, obj, info, filename=None):
        if filename is None:
            hashfn = sha256()
            obj_str = 'None' if obj is None else self.object_formatter(obj)
            dimensions = self._dim_formatter(obj)
            dimensions = dimensions if dimensions else ''
            hashfn.update(obj_str.encode('utf-8'))
            label = sanitizer(getattr(obj, 'label', 'no-label'))
            group = sanitizer(getattr(obj, 'group', 'no-group'))
            format_values = {'timestamp': '{timestamp}', 'dimensions': dimensions, 'group': group, 'label': label, 'type': obj.__class__.__name__, 'obj': sanitizer(obj_str), 'SHA': hashfn.hexdigest()}
            filename = self._format(self.filename_formatter, dict(info, **format_values))
        filename = self._normalize_name(filename)
        ext = info.get('file-ext', '')
        unique_key, ext = self._unique_name(filename, ext, self._files.keys(), force=True)
        return (unique_key, ext)

    def _zip_archive(self, export_name, files, root):
        archname = '.'.join(self._unique_name(export_name, 'zip', root))
        with zipfile.ZipFile(os.path.join(root, archname), 'w') as zipf:
            for (basename, ext), entry in files:
                filename = self._truncate_name(basename, ext)
                zipf.writestr(f'{export_name}/{filename}', Exporter.encode(entry))

    def _tar_archive(self, export_name, files, root):
        archname = '.'.join(self._unique_name(export_name, 'tar', root))
        with tarfile.TarFile(os.path.join(root, archname), 'w') as tarf:
            for (basename, ext), entry in files:
                filename = self._truncate_name(basename, ext)
                tarinfo = tarfile.TarInfo(f'{export_name}/{filename}')
                filedata = Exporter.encode(entry)
                tarinfo.size = len(filedata)
                tarf.addfile(tarinfo, BytesIO(filedata))

    def _single_file_archive(self, export_name, files, root):
        (basename, ext), entry = files[0]
        full_fname = f'{export_name}_{basename}'
        unique_name, ext = self._unique_name(full_fname, ext, root)
        filename = self._truncate_name(self._normalize_name(unique_name), ext=ext)
        fpath = os.path.join(root, filename)
        with open(fpath, 'wb') as f:
            f.write(Exporter.encode(entry))

    def _directory_archive(self, export_name, files, root):
        output_dir = os.path.join(root, self._unique_name(export_name, '', root)[0])
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        for (basename, ext), entry in files:
            filename = self._truncate_name(basename, ext)
            fpath = os.path.join(output_dir, filename)
            with open(fpath, 'wb') as f:
                f.write(Exporter.encode(entry))

    def _unique_name(self, basename, ext, existing, force=False):
        """
        Find a unique basename for a new file/key where existing is
        either a list of (basename, ext) pairs or an absolute path to
        a directory.

        By default, uniqueness is enforced depending on the state of
        the unique_name parameter (for export names). If force is
        True, this parameter is ignored and uniqueness is guaranteed.
        """
        skip = False if force else not self.unique_name
        if skip:
            return (basename, ext)
        ext = '' if ext is None else ext
        if isinstance(existing, str):
            split = [os.path.splitext(el) for el in os.listdir(os.path.abspath(existing))]
            existing = [(n, ex if not ex else ex[1:]) for n, ex in split]
        new_name, counter = (basename, 1)
        while (new_name, ext) in existing:
            new_name = basename + '-' + str(counter)
            counter += 1
        return (sanitizer(new_name), ext)

    def _truncate_name(self, basename, ext='', tail=10, join='...', maxlen=None):
        maxlen = self.max_filename if maxlen is None else maxlen
        max_len = maxlen - len(ext)
        if len(basename) > max_len:
            start = basename[:max_len - (tail + len(join))]
            end = basename[-tail:]
            basename = start + join + end
        filename = f'{basename}.{ext}' if ext else basename
        return filename

    def _normalize_name(self, basename):
        basename = re.sub('-+', '-', basename)
        basename = re.sub('^[-,_]', '', basename)
        return basename.replace(' ', '_')

    def export(self, timestamp=None, info=None):
        """
        Export the archive, directory or file.
        """
        if info is None:
            info = {}
        tval = tuple(time.localtime()) if timestamp is None else timestamp
        tstamp = time.strftime(self.timestamp_format, tval)
        info = dict(info, timestamp=tstamp)
        export_name = self._format(self.export_name, info)
        files = [((self._format(base, info), ext), val) for (base, ext), val in self._files.items()]
        root = os.path.abspath(self.root)
        if len(self) > 1 and (not self.pack):
            self._directory_archive(export_name, files, root)
        elif len(files) == 1:
            self._single_file_archive(export_name, files, root)
        elif self.archive_format == 'zip':
            self._zip_archive(export_name, files, root)
        elif self.archive_format == 'tar':
            self._tar_archive(export_name, files, root)
        if self.flush_archive:
            self._files = {}

    def _format(self, formatter, info):
        filtered = {k: v for k, v in info.items() if k in self.parse_fields(formatter)}
        return formatter.format(**filtered)

    def __len__(self):
        """The number of files currently specified in the archive"""
        return len(self._files)

    def __repr__(self):
        return self.param.pprint()

    def contents(self, maxlen=70):
        """Print the current (unexported) contents of the archive"""
        lines = []
        if len(self._files) == 0:
            print(f'Empty {self.__class__.__name__}')
            return
        fnames = [self._truncate_name(*k, maxlen=maxlen) for k in self._files]
        max_len = max([len(f) for f in fnames])
        for name, v in zip(fnames, self._files.values()):
            mime_type = v[1].get('mime_type', 'no mime type')
            lines.append(f'{name.ljust(max_len)} : {mime_type}')
        print('\n'.join(lines))

    def listing(self):
        """Return a list of filename entries currently in the archive"""
        return [f'{f}.{ext}' if ext else f for f, ext in self._files.keys()]

    def clear(self):
        """Clears the file archive"""
        self._files.clear()