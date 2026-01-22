from fontTools.config import Config
from fontTools.misc import xmlWriter
from fontTools.misc.configTools import AbstractConfig
from fontTools.misc.textTools import Tag, byteord, tostr
from fontTools.misc.loggingTools import deprecateArgument
from fontTools.ttLib import TTLibError
from fontTools.ttLib.ttGlyphSet import _TTGlyph, _TTGlyphSetCFF, _TTGlyphSetGlyf
from fontTools.ttLib.sfnt import SFNTReader, SFNTWriter
from io import BytesIO, StringIO, UnsupportedOperation
import os
import logging
import traceback
class TTFont(object):
    """Represents a TrueType font.

    The object manages file input and output, and offers a convenient way of
    accessing tables. Tables will be only decompiled when necessary, ie. when
    they're actually accessed. This means that simple operations can be extremely fast.

    Example usage::

            >> from fontTools import ttLib
            >> tt = ttLib.TTFont("afont.ttf") # Load an existing font file
            >> tt['maxp'].numGlyphs
            242
            >> tt['OS/2'].achVendID
            'B&H\x00'
            >> tt['head'].unitsPerEm
            2048

    For details of the objects returned when accessing each table, see :ref:`tables`.
    To add a table to the font, use the :py:func:`newTable` function::

            >> os2 = newTable("OS/2")
            >> os2.version = 4
            >> # set other attributes
            >> font["OS/2"] = os2

    TrueType fonts can also be serialized to and from XML format (see also the
    :ref:`ttx` binary)::

            >> tt.saveXML("afont.ttx")
            Dumping 'LTSH' table...
            Dumping 'OS/2' table...
            [...]

            >> tt2 = ttLib.TTFont() # Create a new font object
            >> tt2.importXML("afont.ttx")
            >> tt2['maxp'].numGlyphs
            242

    The TTFont object may be used as a context manager; this will cause the file
    reader to be closed after the context ``with`` block is exited::

            with TTFont(filename) as f:
                    # Do stuff

    Args:
            file: When reading a font from disk, either a pathname pointing to a file,
                    or a readable file object.
            res_name_or_index: If running on a Macintosh, either a sfnt resource name or
                    an sfnt resource index number. If the index number is zero, TTLib will
                    autodetect whether the file is a flat file or a suitcase. (If it is a suitcase,
                    only the first 'sfnt' resource will be read.)
            sfntVersion (str): When constructing a font object from scratch, sets the four-byte
                    sfnt magic number to be used. Defaults to ``\x00\x01\x00\x00`` (TrueType). To create
                    an OpenType file, use ``OTTO``.
            flavor (str): Set this to ``woff`` when creating a WOFF file or ``woff2`` for a WOFF2
                    file.
            checkChecksums (int): How checksum data should be treated. Default is 0
                    (no checking). Set to 1 to check and warn on wrong checksums; set to 2 to
                    raise an exception if any wrong checksums are found.
            recalcBBoxes (bool): If true (the default), recalculates ``glyf``, ``CFF ``,
                    ``head`` bounding box values and ``hhea``/``vhea`` min/max values on save.
                    Also compiles the glyphs on importing, which saves memory consumption and
                    time.
            ignoreDecompileErrors (bool): If true, exceptions raised during table decompilation
                    will be ignored, and the binary data will be returned for those tables instead.
            recalcTimestamp (bool): If true (the default), sets the ``modified`` timestamp in
                    the ``head`` table on save.
            fontNumber (int): The index of the font in a TrueType Collection file.
            lazy (bool): If lazy is set to True, many data structures are loaded lazily, upon
                    access only. If it is set to False, many data structures are loaded immediately.
                    The default is ``lazy=None`` which is somewhere in between.
    """

    def __init__(self, file=None, res_name_or_index=None, sfntVersion='\x00\x01\x00\x00', flavor=None, checkChecksums=0, verbose=None, recalcBBoxes=True, allowVID=NotImplemented, ignoreDecompileErrors=False, recalcTimestamp=True, fontNumber=-1, lazy=None, quiet=None, _tableCache=None, cfg={}):
        for name in ('verbose', 'quiet'):
            val = locals().get(name)
            if val is not None:
                deprecateArgument(name, 'configure logging instead')
            setattr(self, name, val)
        self.lazy = lazy
        self.recalcBBoxes = recalcBBoxes
        self.recalcTimestamp = recalcTimestamp
        self.tables = {}
        self.reader = None
        self.cfg = cfg.copy() if isinstance(cfg, AbstractConfig) else Config(cfg)
        self.ignoreDecompileErrors = ignoreDecompileErrors
        if not file:
            self.sfntVersion = sfntVersion
            self.flavor = flavor
            self.flavorData = None
            return
        seekable = True
        if not hasattr(file, 'read'):
            closeStream = True
            if res_name_or_index is not None:
                from . import macUtils
                if res_name_or_index == 0:
                    if macUtils.getSFNTResIndices(file):
                        file = macUtils.SFNTResourceReader(file, 1)
                    else:
                        file = open(file, 'rb')
                else:
                    file = macUtils.SFNTResourceReader(file, res_name_or_index)
            else:
                file = open(file, 'rb')
        else:
            closeStream = False
            if hasattr(file, 'seekable'):
                seekable = file.seekable()
            elif hasattr(file, 'seek'):
                try:
                    file.seek(0)
                except UnsupportedOperation:
                    seekable = False
        if not self.lazy:
            if seekable:
                file.seek(0)
            tmp = BytesIO(file.read())
            if hasattr(file, 'name'):
                tmp.name = file.name
            if closeStream:
                file.close()
            file = tmp
        elif not seekable:
            raise TTLibError('Input file must be seekable when lazy=True')
        self._tableCache = _tableCache
        self.reader = SFNTReader(file, checkChecksums, fontNumber=fontNumber)
        self.sfntVersion = self.reader.sfntVersion
        self.flavor = self.reader.flavor
        self.flavorData = self.reader.flavorData

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """If we still have a reader object, close it."""
        if self.reader is not None:
            self.reader.close()

    def save(self, file, reorderTables=True):
        """Save the font to disk.

        Args:
                file: Similarly to the constructor, can be either a pathname or a writable
                        file object.
                reorderTables (Option[bool]): If true (the default), reorder the tables,
                        sorting them by tag (recommended by the OpenType specification). If
                        false, retain the original font order. If None, reorder by table
                        dependency (fastest).
        """
        if not hasattr(file, 'write'):
            if self.lazy and self.reader.file.name == file:
                raise TTLibError("Can't overwrite TTFont when 'lazy' attribute is True")
            createStream = True
        else:
            createStream = False
        tmp = BytesIO()
        writer_reordersTables = self._save(tmp)
        if not (reorderTables is None or writer_reordersTables or (reorderTables is False and self.reader is None)):
            if reorderTables is False:
                tableOrder = list(self.reader.keys())
            else:
                tableOrder = None
            tmp.flush()
            tmp2 = BytesIO()
            reorderFontTables(tmp, tmp2, tableOrder)
            tmp.close()
            tmp = tmp2
        if createStream:
            with open(file, 'wb') as file:
                file.write(tmp.getvalue())
        else:
            file.write(tmp.getvalue())
        tmp.close()

    def _save(self, file, tableCache=None):
        """Internal function, to be shared by save() and TTCollection.save()"""
        if self.recalcTimestamp and 'head' in self:
            self['head']
        tags = list(self.keys())
        if 'GlyphOrder' in tags:
            tags.remove('GlyphOrder')
        numTables = len(tags)
        writer = SFNTWriter(file, numTables, self.sfntVersion, self.flavor, self.flavorData)
        done = []
        for tag in tags:
            self._writeTable(tag, writer, done, tableCache)
        writer.close()
        return writer.reordersTables()

    def saveXML(self, fileOrPath, newlinestr='\n', **kwargs):
        """Export the font as TTX (an XML-based text file), or as a series of text
        files when splitTables is true. In the latter case, the 'fileOrPath'
        argument should be a path to a directory.
        The 'tables' argument must either be false (dump all tables) or a
        list of tables to dump. The 'skipTables' argument may be a list of tables
        to skip, but only when the 'tables' argument is false.
        """
        writer = xmlWriter.XMLWriter(fileOrPath, newlinestr=newlinestr)
        self._saveXML(writer, **kwargs)
        writer.close()

    def _saveXML(self, writer, writeVersion=True, quiet=None, tables=None, skipTables=None, splitTables=False, splitGlyphs=False, disassembleInstructions=True, bitmapGlyphDataFormat='raw'):
        if quiet is not None:
            deprecateArgument('quiet', 'configure logging instead')
        self.disassembleInstructions = disassembleInstructions
        self.bitmapGlyphDataFormat = bitmapGlyphDataFormat
        if not tables:
            tables = list(self.keys())
            if 'GlyphOrder' not in tables:
                tables = ['GlyphOrder'] + tables
            if skipTables:
                for tag in skipTables:
                    if tag in tables:
                        tables.remove(tag)
        numTables = len(tables)
        if writeVersion:
            from fontTools import version
            version = '.'.join(version.split('.')[:2])
            writer.begintag('ttFont', sfntVersion=repr(tostr(self.sfntVersion))[1:-1], ttLibVersion=version)
        else:
            writer.begintag('ttFont', sfntVersion=repr(tostr(self.sfntVersion))[1:-1])
        writer.newline()
        splitTables = splitTables or splitGlyphs
        if not splitTables:
            writer.newline()
        else:
            path, ext = os.path.splitext(writer.filename)
        for i in range(numTables):
            tag = tables[i]
            if splitTables:
                tablePath = path + '.' + tagToIdentifier(tag) + ext
                tableWriter = xmlWriter.XMLWriter(tablePath, newlinestr=writer.newlinestr)
                tableWriter.begintag('ttFont', ttLibVersion=version)
                tableWriter.newline()
                tableWriter.newline()
                writer.simpletag(tagToXML(tag), src=os.path.basename(tablePath))
                writer.newline()
            else:
                tableWriter = writer
            self._tableToXML(tableWriter, tag, splitGlyphs=splitGlyphs)
            if splitTables:
                tableWriter.endtag('ttFont')
                tableWriter.newline()
                tableWriter.close()
        writer.endtag('ttFont')
        writer.newline()

    def _tableToXML(self, writer, tag, quiet=None, splitGlyphs=False):
        if quiet is not None:
            deprecateArgument('quiet', 'configure logging instead')
        if tag in self:
            table = self[tag]
            report = "Dumping '%s' table..." % tag
        else:
            report = "No '%s' table found." % tag
        log.info(report)
        if tag not in self:
            return
        xmlTag = tagToXML(tag)
        attrs = dict()
        if hasattr(table, 'ERROR'):
            attrs['ERROR'] = 'decompilation error'
        from .tables.DefaultTable import DefaultTable
        if table.__class__ == DefaultTable:
            attrs['raw'] = True
        writer.begintag(xmlTag, **attrs)
        writer.newline()
        if tag == 'glyf':
            table.toXML(writer, self, splitGlyphs=splitGlyphs)
        else:
            table.toXML(writer, self)
        writer.endtag(xmlTag)
        writer.newline()
        writer.newline()

    def importXML(self, fileOrPath, quiet=None):
        """Import a TTX file (an XML-based text format), so as to recreate
        a font object.
        """
        if quiet is not None:
            deprecateArgument('quiet', 'configure logging instead')
        if 'maxp' in self and 'post' in self:
            self.getGlyphOrder()
        from fontTools.misc import xmlReader
        reader = xmlReader.XMLReader(fileOrPath, self)
        reader.read()

    def isLoaded(self, tag):
        """Return true if the table identified by ``tag`` has been
        decompiled and loaded into memory."""
        return tag in self.tables

    def has_key(self, tag):
        """Test if the table identified by ``tag`` is present in the font.

        As well as this method, ``tag in font`` can also be used to determine the
        presence of the table."""
        if self.isLoaded(tag):
            return True
        elif self.reader and tag in self.reader:
            return True
        elif tag == 'GlyphOrder':
            return True
        else:
            return False
    __contains__ = has_key

    def keys(self):
        """Returns the list of tables in the font, along with the ``GlyphOrder`` pseudo-table."""
        keys = list(self.tables.keys())
        if self.reader:
            for key in list(self.reader.keys()):
                if key not in keys:
                    keys.append(key)
        if 'GlyphOrder' in keys:
            keys.remove('GlyphOrder')
        keys = sortedTagList(keys)
        return ['GlyphOrder'] + keys

    def ensureDecompiled(self, recurse=None):
        """Decompile all the tables, even if a TTFont was opened in 'lazy' mode."""
        for tag in self.keys():
            table = self[tag]
            if recurse is None:
                recurse = self.lazy is not False
            if recurse and hasattr(table, 'ensureDecompiled'):
                table.ensureDecompiled(recurse=recurse)
        self.lazy = False

    def __len__(self):
        return len(list(self.keys()))

    def __getitem__(self, tag):
        tag = Tag(tag)
        table = self.tables.get(tag)
        if table is None:
            if tag == 'GlyphOrder':
                table = GlyphOrder(tag)
                self.tables[tag] = table
            elif self.reader is not None:
                table = self._readTable(tag)
            else:
                raise KeyError("'%s' table not found" % tag)
        return table

    def _readTable(self, tag):
        log.debug("Reading '%s' table from disk", tag)
        data = self.reader[tag]
        if self._tableCache is not None:
            table = self._tableCache.get((tag, data))
            if table is not None:
                return table
        tableClass = getTableClass(tag)
        table = tableClass(tag)
        self.tables[tag] = table
        log.debug("Decompiling '%s' table", tag)
        try:
            table.decompile(data, self)
        except Exception:
            if not self.ignoreDecompileErrors:
                raise
            log.exception("An exception occurred during the decompilation of the '%s' table", tag)
            from .tables.DefaultTable import DefaultTable
            file = StringIO()
            traceback.print_exc(file=file)
            table = DefaultTable(tag)
            table.ERROR = file.getvalue()
            self.tables[tag] = table
            table.decompile(data, self)
        if self._tableCache is not None:
            self._tableCache[tag, data] = table
        return table

    def __setitem__(self, tag, table):
        self.tables[Tag(tag)] = table

    def __delitem__(self, tag):
        if tag not in self:
            raise KeyError("'%s' table not found" % tag)
        if tag in self.tables:
            del self.tables[tag]
        if self.reader and tag in self.reader:
            del self.reader[tag]

    def get(self, tag, default=None):
        """Returns the table if it exists or (optionally) a default if it doesn't."""
        try:
            return self[tag]
        except KeyError:
            return default

    def setGlyphOrder(self, glyphOrder):
        """Set the glyph order

        Args:
                glyphOrder ([str]): List of glyph names in order.
        """
        self.glyphOrder = glyphOrder
        if hasattr(self, '_reverseGlyphOrderDict'):
            del self._reverseGlyphOrderDict
        if self.isLoaded('glyf'):
            self['glyf'].setGlyphOrder(glyphOrder)

    def getGlyphOrder(self):
        """Returns a list of glyph names ordered by their position in the font."""
        try:
            return self.glyphOrder
        except AttributeError:
            pass
        if 'CFF ' in self:
            cff = self['CFF ']
            self.glyphOrder = cff.getGlyphOrder()
        elif 'post' in self:
            glyphOrder = self['post'].getGlyphOrder()
            if glyphOrder is None:
                self._getGlyphNamesFromCmap()
            elif len(glyphOrder) < self['maxp'].numGlyphs:
                log.warning("Not enough names found in the 'post' table, generating them from cmap instead")
                self._getGlyphNamesFromCmap()
            else:
                self.glyphOrder = glyphOrder
        else:
            self._getGlyphNamesFromCmap()
        return self.glyphOrder

    def _getGlyphNamesFromCmap(self):
        if self.isLoaded('cmap'):
            cmapLoading = self.tables['cmap']
            del self.tables['cmap']
        else:
            cmapLoading = None
        numGlyphs = int(self['maxp'].numGlyphs)
        glyphOrder = [None] * numGlyphs
        glyphOrder[0] = '.notdef'
        for i in range(1, numGlyphs):
            glyphOrder[i] = 'glyph%.5d' % i
        self.glyphOrder = glyphOrder
        if 'cmap' in self:
            reversecmap = self['cmap'].buildReversed()
        else:
            reversecmap = {}
        useCount = {}
        for i in range(numGlyphs):
            tempName = glyphOrder[i]
            if tempName in reversecmap:
                glyphName = self._makeGlyphName(min(reversecmap[tempName]))
                numUses = useCount[glyphName] = useCount.get(glyphName, 0) + 1
                if numUses > 1:
                    glyphName = '%s.alt%d' % (glyphName, numUses - 1)
                glyphOrder[i] = glyphName
        if 'cmap' in self:
            del self.tables['cmap']
            self.glyphOrder = glyphOrder
            if cmapLoading:
                self.tables['cmap'] = cmapLoading

    @staticmethod
    def _makeGlyphName(codepoint):
        from fontTools import agl
        if codepoint in agl.UV2AGL:
            return agl.UV2AGL[codepoint]
        elif codepoint <= 65535:
            return 'uni%04X' % codepoint
        else:
            return 'u%X' % codepoint

    def getGlyphNames(self):
        """Get a list of glyph names, sorted alphabetically."""
        glyphNames = sorted(self.getGlyphOrder())
        return glyphNames

    def getGlyphNames2(self):
        """Get a list of glyph names, sorted alphabetically,
        but not case sensitive.
        """
        from fontTools.misc import textTools
        return textTools.caselessSort(self.getGlyphOrder())

    def getGlyphName(self, glyphID):
        """Returns the name for the glyph with the given ID.

        If no name is available, synthesises one with the form ``glyphXXXXX``` where
        ```XXXXX`` is the zero-padded glyph ID.
        """
        try:
            return self.getGlyphOrder()[glyphID]
        except IndexError:
            return 'glyph%.5d' % glyphID

    def getGlyphNameMany(self, lst):
        """Converts a list of glyph IDs into a list of glyph names."""
        glyphOrder = self.getGlyphOrder()
        cnt = len(glyphOrder)
        return [glyphOrder[gid] if gid < cnt else 'glyph%.5d' % gid for gid in lst]

    def getGlyphID(self, glyphName):
        """Returns the ID of the glyph with the given name."""
        try:
            return self.getReverseGlyphMap()[glyphName]
        except KeyError:
            if glyphName[:5] == 'glyph':
                try:
                    return int(glyphName[5:])
                except (NameError, ValueError):
                    raise KeyError(glyphName)
            raise

    def getGlyphIDMany(self, lst):
        """Converts a list of glyph names into a list of glyph IDs."""
        d = self.getReverseGlyphMap()
        try:
            return [d[glyphName] for glyphName in lst]
        except KeyError:
            getGlyphID = self.getGlyphID
            return [getGlyphID(glyphName) for glyphName in lst]

    def getReverseGlyphMap(self, rebuild=False):
        """Returns a mapping of glyph names to glyph IDs."""
        if rebuild or not hasattr(self, '_reverseGlyphOrderDict'):
            self._buildReverseGlyphOrderDict()
        return self._reverseGlyphOrderDict

    def _buildReverseGlyphOrderDict(self):
        self._reverseGlyphOrderDict = d = {}
        for glyphID, glyphName in enumerate(self.getGlyphOrder()):
            d[glyphName] = glyphID
        return d

    def _writeTable(self, tag, writer, done, tableCache=None):
        """Internal helper function for self.save(). Keeps track of
        inter-table dependencies.
        """
        if tag in done:
            return
        tableClass = getTableClass(tag)
        for masterTable in tableClass.dependencies:
            if masterTable not in done:
                if masterTable in self:
                    self._writeTable(masterTable, writer, done, tableCache)
                else:
                    done.append(masterTable)
        done.append(tag)
        tabledata = self.getTableData(tag)
        if tableCache is not None:
            entry = tableCache.get((Tag(tag), tabledata))
            if entry is not None:
                log.debug("reusing '%s' table", tag)
                writer.setEntry(tag, entry)
                return
        log.debug("Writing '%s' table to disk", tag)
        writer[tag] = tabledata
        if tableCache is not None:
            tableCache[Tag(tag), tabledata] = writer[tag]

    def getTableData(self, tag):
        """Returns the binary representation of a table.

        If the table is currently loaded and in memory, the data is compiled to
        binary and returned; if it is not currently loaded, the binary data is
        read from the font file and returned.
        """
        tag = Tag(tag)
        if self.isLoaded(tag):
            log.debug("Compiling '%s' table", tag)
            return self.tables[tag].compile(self)
        elif self.reader and tag in self.reader:
            log.debug("Reading '%s' table from disk", tag)
            return self.reader[tag]
        else:
            raise KeyError(tag)

    def getGlyphSet(self, preferCFF=True, location=None, normalized=False, recalcBounds=True):
        """Return a generic GlyphSet, which is a dict-like object
        mapping glyph names to glyph objects. The returned glyph objects
        have a ``.draw()`` method that supports the Pen protocol, and will
        have an attribute named 'width'.

        If the font is CFF-based, the outlines will be taken from the ``CFF ``
        or ``CFF2`` tables. Otherwise the outlines will be taken from the
        ``glyf`` table.

        If the font contains both a ``CFF ``/``CFF2`` and a ``glyf`` table, you
        can use the ``preferCFF`` argument to specify which one should be taken.
        If the font contains both a ``CFF `` and a ``CFF2`` table, the latter is
        taken.

        If the ``location`` parameter is set, it should be a dictionary mapping
        four-letter variation tags to their float values, and the returned
        glyph-set will represent an instance of a variable font at that
        location.

        If the ``normalized`` variable is set to True, that location is
        interpreted as in the normalized (-1..+1) space, otherwise it is in the
        font's defined axes space.
        """
        if location and 'fvar' not in self:
            location = None
        if location and (not normalized):
            location = self.normalizeLocation(location)
        if ('CFF ' in self or 'CFF2' in self) and (preferCFF or 'glyf' not in self):
            return _TTGlyphSetCFF(self, location)
        elif 'glyf' in self:
            return _TTGlyphSetGlyf(self, location, recalcBounds=recalcBounds)
        else:
            raise TTLibError('Font contains no outlines')

    def normalizeLocation(self, location):
        """Normalize a ``location`` from the font's defined axes space (also
        known as user space) into the normalized (-1..+1) space. It applies
        ``avar`` mapping if the font contains an ``avar`` table.

        The ``location`` parameter should be a dictionary mapping four-letter
        variation tags to their float values.

        Raises ``TTLibError`` if the font is not a variable font.
        """
        from fontTools.varLib.models import normalizeLocation, piecewiseLinearMap
        if 'fvar' not in self:
            raise TTLibError('Not a variable font')
        axes = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in self['fvar'].axes}
        location = normalizeLocation(location, axes)
        if 'avar' in self:
            avar = self['avar']
            avarSegments = avar.segments
            mappedLocation = {}
            for axisTag, value in location.items():
                avarMapping = avarSegments.get(axisTag, None)
                if avarMapping is not None:
                    value = piecewiseLinearMap(value, avarMapping)
                mappedLocation[axisTag] = value
            location = mappedLocation
        return location

    def getBestCmap(self, cmapPreferences=((3, 10), (0, 6), (0, 4), (3, 1), (0, 3), (0, 2), (0, 1), (0, 0))):
        """Returns the 'best' Unicode cmap dictionary available in the font
        or ``None``, if no Unicode cmap subtable is available.

        By default it will search for the following (platformID, platEncID)
        pairs in order::

                        (3, 10), # Windows Unicode full repertoire
                        (0, 6),  # Unicode full repertoire (format 13 subtable)
                        (0, 4),  # Unicode 2.0 full repertoire
                        (3, 1),  # Windows Unicode BMP
                        (0, 3),  # Unicode 2.0 BMP
                        (0, 2),  # Unicode ISO/IEC 10646
                        (0, 1),  # Unicode 1.1
                        (0, 0)   # Unicode 1.0

        This particular order matches what HarfBuzz uses to choose what
        subtable to use by default. This order prefers the largest-repertoire
        subtable, and among those, prefers the Windows-platform over the
        Unicode-platform as the former has wider support.

        This order can be customized via the ``cmapPreferences`` argument.
        """
        return self['cmap'].getBestCmap(cmapPreferences=cmapPreferences)