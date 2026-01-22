import os
from copy import deepcopy
from os import fsdecode
import logging
import zipfile
import enum
from collections import OrderedDict
import fs
import fs.base
import fs.subfs
import fs.errors
import fs.copy
import fs.osfs
import fs.zipfs
import fs.tempfs
import fs.tools
from fontTools.misc import plistlib
from fontTools.ufoLib.validators import *
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.converters import convertUFO1OrUFO2KerningToUFO3Kerning
from fontTools.ufoLib.errors import UFOLibError
from fontTools.ufoLib.utils import numberTypes, _VersionTupleEnumMixin
class UFOWriter(UFOReader):
    """
    Write the various components of the .ufo.

    By default, the written data will be validated before writing. Set ``validate`` to
    ``False`` if you do not want to validate the data. Validation can also be overriden
    on a per method level if desired.

    The ``formatVersion`` argument allows to specify the UFO format version as a tuple
    of integers (major, minor), or as a single integer for the major digit only (minor
    is implied as 0). By default the latest formatVersion will be used; currently it's
    3.0, which is equivalent to formatVersion=(3, 0).

    An UnsupportedUFOFormat exception is raised if the requested UFO formatVersion is
    not supported.
    """

    def __init__(self, path, formatVersion=None, fileCreator='com.github.fonttools.ufoLib', structure=None, validate=True):
        try:
            formatVersion = UFOFormatVersion(formatVersion)
        except ValueError as e:
            from fontTools.ufoLib.errors import UnsupportedUFOFormat
            raise UnsupportedUFOFormat(f'Unsupported UFO format: {formatVersion!r}') from e
        if hasattr(path, '__fspath__'):
            path = path.__fspath__()
        if isinstance(path, str):
            path = os.path.normpath(path)
            havePreviousFile = os.path.exists(path)
            if havePreviousFile:
                existingStructure = _sniffFileStructure(path)
                if structure is not None:
                    try:
                        structure = UFOFileStructure(structure)
                    except ValueError:
                        raise UFOLibError("Invalid or unsupported structure: '%s'" % structure)
                    if structure is not existingStructure:
                        raise UFOLibError("A UFO with a different structure (%s) already exists at the given path: '%s'" % (existingStructure, path))
                else:
                    structure = existingStructure
            else:
                if structure is None:
                    structure = UFOFileStructure.PACKAGE
                dirName = os.path.dirname(path)
                if dirName and (not os.path.isdir(dirName)):
                    raise UFOLibError("Cannot write to '%s': directory does not exist" % path)
            if structure is UFOFileStructure.ZIP:
                if havePreviousFile:
                    parentFS = fs.tempfs.TempFS()
                    with fs.zipfs.ZipFS(path, encoding='utf-8') as origFS:
                        fs.copy.copy_fs(origFS, parentFS)
                    rootDirs = [p.name for p in parentFS.scandir('/') if p.is_dir and p.name != '__MACOSX']
                    if len(rootDirs) != 1:
                        raise UFOLibError('Expected exactly 1 root directory, found %d' % len(rootDirs))
                    else:
                        self.fs = parentFS.opendir(rootDirs[0], factory=fs.subfs.ClosingSubFS)
                else:
                    rootDir = os.path.splitext(os.path.basename(path))[0] + '.ufo'
                    parentFS = fs.zipfs.ZipFS(path, write=True, encoding='utf-8')
                    parentFS.makedir(rootDir)
                    self.fs = parentFS.opendir(rootDir, factory=fs.subfs.ClosingSubFS)
            else:
                self.fs = fs.osfs.OSFS(path, create=True)
            self._fileStructure = structure
            self._havePreviousFile = havePreviousFile
            self._shouldClose = True
        elif isinstance(path, fs.base.FS):
            filesystem = path
            try:
                filesystem.check()
            except fs.errors.FilesystemClosed:
                raise UFOLibError("the filesystem '%s' is closed" % path)
            else:
                self.fs = filesystem
            try:
                path = filesystem.getsyspath('/')
            except fs.errors.NoSysPath:
                path = str(filesystem)
            if structure and structure is not UFOFileStructure.PACKAGE:
                import warnings
                warnings.warn("The 'structure' argument is not used when input is an FS object", UserWarning, stacklevel=2)
            self._fileStructure = UFOFileStructure.PACKAGE
            self._havePreviousFile = filesystem.exists(METAINFO_FILENAME)
            self._shouldClose = False
        else:
            raise TypeError('Expected a path string or fs object, found %s' % type(path).__name__)
        self._path = fsdecode(path)
        self._formatVersion = formatVersion
        self._fileCreator = fileCreator
        self._downConversionKerningData = None
        self._validate = validate
        previousFormatVersion = None
        if self._havePreviousFile:
            metaInfo = self._readMetaInfo(validate=validate)
            previousFormatVersion = metaInfo['formatVersionTuple']
            if previousFormatVersion > formatVersion:
                from fontTools.ufoLib.errors import UnsupportedUFOFormat
                raise UnsupportedUFOFormat(f'The UFO located at this path is a higher version ({previousFormatVersion}) than the version ({formatVersion}) that is trying to be written. This is not supported.')
        self.layerContents = {}
        if previousFormatVersion is not None and previousFormatVersion.major >= 3:
            self.layerContents = OrderedDict(self._readLayerContents(validate))
        elif self.fs.exists(DEFAULT_GLYPHS_DIRNAME):
            self.layerContents = {DEFAULT_LAYER_NAME: DEFAULT_GLYPHS_DIRNAME}
        self._writeMetaInfo()

    def _get_fileCreator(self):
        return self._fileCreator
    fileCreator = property(_get_fileCreator, doc='The file creator of the UFO. This is set into metainfo.plist during __init__.')

    def copyFromReader(self, reader, sourcePath, destPath):
        """
        Copy the sourcePath in the provided UFOReader to destPath
        in this writer. The paths must be relative. This works with
        both individual files and directories.
        """
        if not isinstance(reader, UFOReader):
            raise UFOLibError('The reader must be an instance of UFOReader.')
        sourcePath = fsdecode(sourcePath)
        destPath = fsdecode(destPath)
        if not reader.fs.exists(sourcePath):
            raise UFOLibError('The reader does not have data located at "%s".' % sourcePath)
        if self.fs.exists(destPath):
            raise UFOLibError('A file named "%s" already exists.' % destPath)
        self.fs.makedirs(fs.path.dirname(destPath), recreate=True)
        if reader.fs.isdir(sourcePath):
            fs.copy.copy_dir(reader.fs, sourcePath, self.fs, destPath)
        else:
            fs.copy.copy_file(reader.fs, sourcePath, self.fs, destPath)

    def writeBytesToPath(self, path, data):
        """
        Write bytes to a path relative to the UFO filesystem's root.
        If writing to an existing UFO, check to see if data matches the data
        that is already in the file at path; if so, the file is not rewritten
        so that the modification date is preserved.
        If needed, the directory tree for the given path will be built.
        """
        path = fsdecode(path)
        if self._havePreviousFile:
            if self.fs.isfile(path) and data == self.fs.readbytes(path):
                return
        try:
            self.fs.writebytes(path, data)
        except fs.errors.FileExpected:
            raise UFOLibError("A directory exists at '%s'" % path)
        except fs.errors.ResourceNotFound:
            self.fs.makedirs(fs.path.dirname(path), recreate=True)
            self.fs.writebytes(path, data)

    def getFileObjectForPath(self, path, mode='w', encoding=None):
        """
        Returns a file (or file-like) object for the
        file at the given path. The path must be relative
        to the UFO path. Returns None if the file does
        not exist and the mode is "r" or "rb.
        An encoding may be passed if the file is opened in text mode.

        Note: The caller is responsible for closing the open file.
        """
        path = fsdecode(path)
        try:
            return self.fs.open(path, mode=mode, encoding=encoding)
        except fs.errors.ResourceNotFound as e:
            m = mode[0]
            if m == 'r':
                return None
            elif m == 'w' or m == 'a' or m == 'x':
                self.fs.makedirs(fs.path.dirname(path), recreate=True)
                return self.fs.open(path, mode=mode, encoding=encoding)
        except fs.errors.ResourceError as e:
            return UFOLibError(f"unable to open '{path}' on {self.fs}: {e}")

    def removePath(self, path, force=False, removeEmptyParents=True):
        """
        Remove the file (or directory) at path. The path
        must be relative to the UFO.
        Raises UFOLibError if the path doesn't exist.
        If force=True, ignore non-existent paths.
        If the directory where 'path' is located becomes empty, it will
        be automatically removed, unless 'removeEmptyParents' is False.
        """
        path = fsdecode(path)
        try:
            self.fs.remove(path)
        except fs.errors.FileExpected:
            self.fs.removetree(path)
        except fs.errors.ResourceNotFound:
            if not force:
                raise UFOLibError(f"'{path}' does not exist on {self.fs}")
        if removeEmptyParents:
            parent = fs.path.dirname(path)
            if parent:
                fs.tools.remove_empty(self.fs, parent)
    removeFileForPath = removePath

    def setModificationTime(self):
        """
        Set the UFO modification time to the current time.
        This is never called automatically. It is up to the
        caller to call this when finished working on the UFO.
        """
        path = self._path
        if path is not None and os.path.exists(path):
            try:
                os.utime(path, None)
            except OSError as e:
                logger.warning('Failed to set modified time: %s', e)

    def _writeMetaInfo(self):
        metaInfo = dict(creator=self._fileCreator, formatVersion=self._formatVersion.major)
        if self._formatVersion.minor != 0:
            metaInfo['formatVersionMinor'] = self._formatVersion.minor
        self._writePlist(METAINFO_FILENAME, metaInfo)

    def setKerningGroupConversionRenameMaps(self, maps):
        """
        Set maps defining the renaming that should be done
        when writing groups and kerning in UFO 1 and UFO 2.
        This will effectively undo the conversion done when
        UFOReader reads this data. The dictionary should have
        this form::

                {
                        "side1" : {"group name to use when writing" : "group name in data"},
                        "side2" : {"group name to use when writing" : "group name in data"}
                }

        This is the same form returned by UFOReader's
        getKerningGroupConversionRenameMaps method.
        """
        if self._formatVersion >= UFOFormatVersion.FORMAT_3_0:
            return
        remap = {}
        for side in ('side1', 'side2'):
            for writeName, dataName in list(maps[side].items()):
                remap[dataName] = writeName
        self._downConversionKerningData = dict(groupRenameMap=remap)

    def writeGroups(self, groups, validate=None):
        """
        Write groups.plist. This method requires a
        dict of glyph groups as an argument.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        if validate:
            valid, message = groupsValidator(groups)
            if not valid:
                raise UFOLibError(message)
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0 and self._downConversionKerningData is not None:
            remap = self._downConversionKerningData['groupRenameMap']
            remappedGroups = {}
            for name, contents in list(groups.items()):
                if name in remap:
                    continue
                remappedGroups[name] = contents
            for name, contents in list(groups.items()):
                if name not in remap:
                    continue
                name = remap[name]
                remappedGroups[name] = contents
            groups = remappedGroups
        groupsNew = {}
        for key, value in groups.items():
            groupsNew[key] = list(value)
        if groupsNew:
            self._writePlist(GROUPS_FILENAME, groupsNew)
        elif self._havePreviousFile:
            self.removePath(GROUPS_FILENAME, force=True, removeEmptyParents=False)

    def writeInfo(self, info, validate=None):
        """
        Write info.plist. This method requires an object
        that supports getting attributes that follow the
        fontinfo.plist version 2 specification. Attributes
        will be taken from the given object and written
        into the file.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        infoData = {}
        for attr in list(fontInfoAttributesVersion3ValueData.keys()):
            if hasattr(info, attr):
                try:
                    value = getattr(info, attr)
                except AttributeError:
                    raise UFOLibError('The supplied info object does not support getting a necessary attribute (%s).' % attr)
                if value is None:
                    continue
                infoData[attr] = value
        if self._formatVersion == UFOFormatVersion.FORMAT_3_0:
            if validate:
                infoData = validateInfoVersion3Data(infoData)
        elif self._formatVersion == UFOFormatVersion.FORMAT_2_0:
            infoData = _convertFontInfoDataVersion3ToVersion2(infoData)
            if validate:
                infoData = validateInfoVersion2Data(infoData)
        elif self._formatVersion == UFOFormatVersion.FORMAT_1_0:
            infoData = _convertFontInfoDataVersion3ToVersion2(infoData)
            if validate:
                infoData = validateInfoVersion2Data(infoData)
            infoData = _convertFontInfoDataVersion2ToVersion1(infoData)
        if infoData:
            self._writePlist(FONTINFO_FILENAME, infoData)

    def writeKerning(self, kerning, validate=None):
        """
        Write kerning.plist. This method requires a
        dict of kerning pairs as an argument.

        This performs basic structural validation of the kerning,
        but it does not check for compliance with the spec in
        regards to conflicting pairs. The assumption is that the
        kerning data being passed is standards compliant.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        if validate:
            invalidFormatMessage = 'The kerning is not properly formatted.'
            if not isDictEnough(kerning):
                raise UFOLibError(invalidFormatMessage)
            for pair, value in list(kerning.items()):
                if not isinstance(pair, (list, tuple)):
                    raise UFOLibError(invalidFormatMessage)
                if not len(pair) == 2:
                    raise UFOLibError(invalidFormatMessage)
                if not isinstance(pair[0], str):
                    raise UFOLibError(invalidFormatMessage)
                if not isinstance(pair[1], str):
                    raise UFOLibError(invalidFormatMessage)
                if not isinstance(value, numberTypes):
                    raise UFOLibError(invalidFormatMessage)
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0 and self._downConversionKerningData is not None:
            remap = self._downConversionKerningData['groupRenameMap']
            remappedKerning = {}
            for (side1, side2), value in list(kerning.items()):
                side1 = remap.get(side1, side1)
                side2 = remap.get(side2, side2)
                remappedKerning[side1, side2] = value
            kerning = remappedKerning
        kerningDict = {}
        for left, right in kerning.keys():
            value = kerning[left, right]
            if left not in kerningDict:
                kerningDict[left] = {}
            kerningDict[left][right] = value
        if kerningDict:
            self._writePlist(KERNING_FILENAME, kerningDict)
        elif self._havePreviousFile:
            self.removePath(KERNING_FILENAME, force=True, removeEmptyParents=False)

    def writeLib(self, libDict, validate=None):
        """
        Write lib.plist. This method requires a
        lib dict as an argument.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        if validate:
            valid, message = fontLibValidator(libDict)
            if not valid:
                raise UFOLibError(message)
        if libDict:
            self._writePlist(LIB_FILENAME, libDict)
        elif self._havePreviousFile:
            self.removePath(LIB_FILENAME, force=True, removeEmptyParents=False)

    def writeFeatures(self, features, validate=None):
        """
        Write features.fea. This method requires a
        features string as an argument.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion == UFOFormatVersion.FORMAT_1_0:
            raise UFOLibError('features.fea is not allowed in UFO Format Version 1.')
        if validate:
            if not isinstance(features, str):
                raise UFOLibError('The features are not text.')
        if features:
            self.writeBytesToPath(FEATURES_FILENAME, features.encode('utf8'))
        elif self._havePreviousFile:
            self.removePath(FEATURES_FILENAME, force=True, removeEmptyParents=False)

    def writeLayerContents(self, layerOrder=None, validate=None):
        """
        Write the layercontents.plist file. This method  *must* be called
        after all glyph sets have been written.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            return
        if layerOrder is not None:
            newOrder = []
            for layerName in layerOrder:
                if layerName is None:
                    layerName = DEFAULT_LAYER_NAME
                newOrder.append(layerName)
            layerOrder = newOrder
        else:
            layerOrder = list(self.layerContents.keys())
        if validate and set(layerOrder) != set(self.layerContents.keys()):
            raise UFOLibError('The layer order content does not match the glyph sets that have been created.')
        layerContents = [(layerName, self.layerContents[layerName]) for layerName in layerOrder]
        self._writePlist(LAYERCONTENTS_FILENAME, layerContents)

    def _findDirectoryForLayerName(self, layerName):
        foundDirectory = None
        for existingLayerName, directoryName in list(self.layerContents.items()):
            if layerName is None and directoryName == DEFAULT_GLYPHS_DIRNAME:
                foundDirectory = directoryName
                break
            elif existingLayerName == layerName:
                foundDirectory = directoryName
                break
        if not foundDirectory:
            raise UFOLibError('Could not locate a glyph set directory for the layer named %s.' % layerName)
        return foundDirectory

    def getGlyphSet(self, layerName=None, defaultLayer=True, glyphNameToFileNameFunc=None, validateRead=None, validateWrite=None, expectContentsFile=False):
        """
        Return the GlyphSet object associated with the
        appropriate glyph directory in the .ufo.
        If layerName is None, the default glyph set
        will be used. The defaultLayer flag indictes
        that the layer should be saved into the default
        glyphs directory.

        ``validateRead`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        ``validateWrte`` will validate the written data, by default it is set to the
        class's validate value, can be overridden.
        ``expectContentsFile`` will raise a GlifLibError if a contents.plist file is
        not found on the glyph set file system. This should be set to ``True`` if you
        are reading an existing UFO and ``False`` if you use ``getGlyphSet`` to create
        a fresh	glyph set.
        """
        if validateRead is None:
            validateRead = self._validate
        if validateWrite is None:
            validateWrite = self._validate
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0 and (not defaultLayer or layerName is not None):
            raise UFOLibError(f'Only the default layer can be writen in UFO {self._formatVersion.major}.')
        if layerName is None and defaultLayer:
            for existingLayerName, directory in self.layerContents.items():
                if directory == DEFAULT_GLYPHS_DIRNAME:
                    layerName = existingLayerName
            if layerName is None:
                layerName = DEFAULT_LAYER_NAME
        elif layerName is None and (not defaultLayer):
            raise UFOLibError('A layer name must be provided for non-default layers.')
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            return self._getDefaultGlyphSet(validateRead, validateWrite, glyphNameToFileNameFunc=glyphNameToFileNameFunc, expectContentsFile=expectContentsFile)
        elif self._formatVersion.major == UFOFormatVersion.FORMAT_3_0.major:
            return self._getGlyphSetFormatVersion3(validateRead, validateWrite, layerName=layerName, defaultLayer=defaultLayer, glyphNameToFileNameFunc=glyphNameToFileNameFunc, expectContentsFile=expectContentsFile)
        else:
            raise NotImplementedError(self._formatVersion)

    def _getDefaultGlyphSet(self, validateRead, validateWrite, glyphNameToFileNameFunc=None, expectContentsFile=False):
        from fontTools.ufoLib.glifLib import GlyphSet
        glyphSubFS = self.fs.makedir(DEFAULT_GLYPHS_DIRNAME, recreate=True)
        return GlyphSet(glyphSubFS, glyphNameToFileNameFunc=glyphNameToFileNameFunc, ufoFormatVersion=self._formatVersion, validateRead=validateRead, validateWrite=validateWrite, expectContentsFile=expectContentsFile)

    def _getGlyphSetFormatVersion3(self, validateRead, validateWrite, layerName=None, defaultLayer=True, glyphNameToFileNameFunc=None, expectContentsFile=False):
        from fontTools.ufoLib.glifLib import GlyphSet
        if defaultLayer:
            for existingLayerName, directory in self.layerContents.items():
                if directory == DEFAULT_GLYPHS_DIRNAME:
                    if existingLayerName != layerName:
                        raise UFOLibError("Another layer ('%s') is already mapped to the default directory." % existingLayerName)
                elif existingLayerName == layerName:
                    raise UFOLibError('The layer name is already mapped to a non-default layer.')
        if layerName in self.layerContents:
            directory = self.layerContents[layerName]
        elif defaultLayer:
            directory = DEFAULT_GLYPHS_DIRNAME
        else:
            existing = {d.lower() for d in self.layerContents.values()}
            directory = userNameToFileName(layerName, existing=existing, prefix='glyphs.')
        glyphSubFS = self.fs.makedir(directory, recreate=True)
        self.layerContents[layerName] = directory
        return GlyphSet(glyphSubFS, glyphNameToFileNameFunc=glyphNameToFileNameFunc, ufoFormatVersion=self._formatVersion, validateRead=validateRead, validateWrite=validateWrite, expectContentsFile=expectContentsFile)

    def renameGlyphSet(self, layerName, newLayerName, defaultLayer=False):
        """
        Rename a glyph set.

        Note: if a GlyphSet object has already been retrieved for
        layerName, it is up to the caller to inform that object that
        the directory it represents has changed.
        """
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            return
        if layerName == newLayerName:
            if self.layerContents[layerName] != DEFAULT_GLYPHS_DIRNAME and (not defaultLayer):
                return
            if self.layerContents[layerName] == DEFAULT_GLYPHS_DIRNAME and defaultLayer:
                return
        else:
            if newLayerName is None:
                newLayerName = DEFAULT_LAYER_NAME
            if newLayerName in self.layerContents:
                raise UFOLibError('A layer named %s already exists.' % newLayerName)
            if defaultLayer and DEFAULT_GLYPHS_DIRNAME in self.layerContents.values():
                raise UFOLibError('A default layer already exists.')
        oldDirectory = self._findDirectoryForLayerName(layerName)
        if defaultLayer:
            newDirectory = DEFAULT_GLYPHS_DIRNAME
        else:
            existing = {name.lower() for name in self.layerContents.values()}
            newDirectory = userNameToFileName(newLayerName, existing=existing, prefix='glyphs.')
        del self.layerContents[layerName]
        self.layerContents[newLayerName] = newDirectory
        self.fs.movedir(oldDirectory, newDirectory, create=True)

    def deleteGlyphSet(self, layerName):
        """
        Remove the glyph set matching layerName.
        """
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            return
        foundDirectory = self._findDirectoryForLayerName(layerName)
        self.removePath(foundDirectory, removeEmptyParents=False)
        del self.layerContents[layerName]

    def writeData(self, fileName, data):
        """
        Write data to fileName in the 'data' directory.
        The data must be a bytes string.
        """
        self.writeBytesToPath(f'{DATA_DIRNAME}/{fsdecode(fileName)}', data)

    def removeData(self, fileName):
        """
        Remove the file named fileName from the data directory.
        """
        self.removePath(f'{DATA_DIRNAME}/{fsdecode(fileName)}')

    def writeImage(self, fileName, data, validate=None):
        """
        Write data to fileName in the images directory.
        The data must be a valid PNG.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            raise UFOLibError(f'Images are not allowed in UFO {self._formatVersion.major}.')
        fileName = fsdecode(fileName)
        if validate:
            valid, error = pngValidator(data=data)
            if not valid:
                raise UFOLibError(error)
        self.writeBytesToPath(f'{IMAGES_DIRNAME}/{fileName}', data)

    def removeImage(self, fileName, validate=None):
        """
        Remove the file named fileName from the
        images directory.
        """
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            raise UFOLibError(f'Images are not allowed in UFO {self._formatVersion.major}.')
        self.removePath(f'{IMAGES_DIRNAME}/{fsdecode(fileName)}')

    def copyImageFromReader(self, reader, sourceFileName, destFileName, validate=None):
        """
        Copy the sourceFileName in the provided UFOReader to destFileName
        in this writer. This uses the most memory efficient method possible
        for copying the data possible.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            raise UFOLibError(f'Images are not allowed in UFO {self._formatVersion.major}.')
        sourcePath = f'{IMAGES_DIRNAME}/{fsdecode(sourceFileName)}'
        destPath = f'{IMAGES_DIRNAME}/{fsdecode(destFileName)}'
        self.copyFromReader(reader, sourcePath, destPath)

    def close(self):
        if self._havePreviousFile and self._fileStructure is UFOFileStructure.ZIP:
            rootDir = os.path.splitext(os.path.basename(self._path))[0] + '.ufo'
            with fs.zipfs.ZipFS(self._path, write=True, encoding='utf-8') as destFS:
                fs.copy.copy_fs(self.fs, destFS.makedir(rootDir))
        super().close()