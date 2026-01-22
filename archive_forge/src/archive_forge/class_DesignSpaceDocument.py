from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
class DesignSpaceDocument(LogMixin, AsDictMixin):
    """The DesignSpaceDocument object can read and write ``.designspace`` data.
    It imports the axes, sources, variable fonts and instances to very basic
    **descriptor** objects that store the data in attributes. Data is added to
    the document by creating such descriptor objects, filling them with data
    and then adding them to the document. This makes it easy to integrate this
    object in different contexts.

    The **DesignSpaceDocument** object can be subclassed to work with
    different objects, as long as they have the same attributes. Reader and
    Writer objects can be subclassed as well.

    **Note:** Python attribute names are usually camelCased, the
    corresponding `XML <document-xml-structure>`_ attributes are usually
    all lowercase.

    .. code:: python

        from fontTools.designspaceLib import DesignSpaceDocument
        doc = DesignSpaceDocument.fromfile("some/path/to/my.designspace")
        doc.formatVersion
        doc.elidedFallbackName
        doc.axes
        doc.axisMappings
        doc.locationLabels
        doc.rules
        doc.rulesProcessingLast
        doc.sources
        doc.variableFonts
        doc.instances
        doc.lib

    """

    def __init__(self, readerClass=None, writerClass=None):
        self.path = None
        'String, optional. When the document is read from the disk, this is\n        the full path that was given to :meth:`read` or :meth:`fromfile`.\n        '
        self.filename = None
        'String, optional. When the document is read from the disk, this is\n        its original file name, i.e. the last part of its path.\n\n        When the document is produced by a Python script and still only exists\n        in memory, the producing script can write here an indication of a\n        possible "good" filename, in case one wants to save the file somewhere.\n        '
        self.formatVersion: Optional[str] = None
        'Format version for this document, as a string. E.g. "4.0" '
        self.elidedFallbackName: Optional[str] = None
        'STAT Style Attributes Header field ``elidedFallbackNameID``.\n\n        See: `OTSpec STAT Style Attributes Header <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#style-attributes-header>`_\n\n        .. versionadded:: 5.0\n        '
        self.axes: List[Union[AxisDescriptor, DiscreteAxisDescriptor]] = []
        "List of this document's axes."
        self.axisMappings: List[AxisMappingDescriptor] = []
        "List of this document's axis mappings."
        self.locationLabels: List[LocationLabelDescriptor] = []
        "List of this document's STAT format 4 labels.\n\n        .. versionadded:: 5.0"
        self.rules: List[RuleDescriptor] = []
        "List of this document's rules."
        self.rulesProcessingLast: bool = False
        'This flag indicates whether the substitution rules should be applied\n        before or after other glyph substitution features.\n\n        - False: before\n        - True: after.\n\n        Default is False. For new projects, you probably want True. See\n        the following issues for more information:\n        `fontTools#1371 <https://github.com/fonttools/fonttools/issues/1371#issuecomment-590214572>`__\n        `fontTools#2050 <https://github.com/fonttools/fonttools/issues/2050#issuecomment-678691020>`__\n\n        If you want to use a different feature altogether, e.g. ``calt``,\n        use the lib key ``com.github.fonttools.varLib.featureVarsFeatureTag``\n\n        .. code:: xml\n\n            <lib>\n                <dict>\n                    <key>com.github.fonttools.varLib.featureVarsFeatureTag</key>\n                    <string>calt</string>\n                </dict>\n            </lib>\n        '
        self.sources: List[SourceDescriptor] = []
        "List of this document's sources."
        self.variableFonts: List[VariableFontDescriptor] = []
        "List of this document's variable fonts.\n\n        .. versionadded:: 5.0"
        self.instances: List[InstanceDescriptor] = []
        "List of this document's instances."
        self.lib: Dict = {}
        'User defined, custom data associated with the whole document.\n\n        Use reverse-DNS notation to identify your own data.\n        Respect the data stored by others.\n        '
        self.default: Optional[str] = None
        'Name of the default master.\n\n        This attribute is updated by the :meth:`findDefault`\n        '
        if readerClass is not None:
            self.readerClass = readerClass
        else:
            self.readerClass = BaseDocReader
        if writerClass is not None:
            self.writerClass = writerClass
        else:
            self.writerClass = BaseDocWriter

    @classmethod
    def fromfile(cls, path, readerClass=None, writerClass=None):
        """Read a designspace file from ``path`` and return a new instance of
        :class:.
        """
        self = cls(readerClass=readerClass, writerClass=writerClass)
        self.read(path)
        return self

    @classmethod
    def fromstring(cls, string, readerClass=None, writerClass=None):
        self = cls(readerClass=readerClass, writerClass=writerClass)
        reader = self.readerClass.fromstring(string, self)
        reader.read()
        if self.sources:
            self.findDefault()
        return self

    def tostring(self, encoding=None):
        """Returns the designspace as a string. Default encoding ``utf-8``."""
        if encoding is str or (encoding is not None and encoding.lower() == 'unicode'):
            f = StringIO()
            xml_declaration = False
        elif encoding is None or encoding == 'utf-8':
            f = BytesIO()
            encoding = 'UTF-8'
            xml_declaration = True
        else:
            raise ValueError("unsupported encoding: '%s'" % encoding)
        writer = self.writerClass(f, self)
        writer.write(encoding=encoding, xml_declaration=xml_declaration)
        return f.getvalue()

    def read(self, path):
        """Read a designspace file from ``path`` and populates the fields of
        ``self`` with the data.
        """
        if hasattr(path, '__fspath__'):
            path = path.__fspath__()
        self.path = path
        self.filename = os.path.basename(path)
        reader = self.readerClass(path, self)
        reader.read()
        if self.sources:
            self.findDefault()

    def write(self, path):
        """Write this designspace to ``path``."""
        if hasattr(path, '__fspath__'):
            path = path.__fspath__()
        self.path = path
        self.filename = os.path.basename(path)
        self.updatePaths()
        writer = self.writerClass(path, self)
        writer.write()

    def _posixRelativePath(self, otherPath):
        relative = os.path.relpath(otherPath, os.path.dirname(self.path))
        return posix(relative)

    def updatePaths(self):
        """
        Right before we save we need to identify and respond to the following situations:
        In each descriptor, we have to do the right thing for the filename attribute.

        ::

            case 1.
            descriptor.filename == None
            descriptor.path == None

            -- action:
            write as is, descriptors will not have a filename attr.
            useless, but no reason to interfere.


            case 2.
            descriptor.filename == "../something"
            descriptor.path == None

            -- action:
            write as is. The filename attr should not be touched.


            case 3.
            descriptor.filename == None
            descriptor.path == "~/absolute/path/there"

            -- action:
            calculate the relative path for filename.
            We're not overwriting some other value for filename, it should be fine


            case 4.
            descriptor.filename == '../somewhere'
            descriptor.path == "~/absolute/path/there"

            -- action:
            there is a conflict between the given filename, and the path.
            So we know where the file is relative to the document.
            Can't guess why they're different, we just choose for path to be correct and update filename.
        """
        assert self.path is not None
        for descriptor in self.sources + self.instances:
            if descriptor.path is not None:
                descriptor.filename = self._posixRelativePath(descriptor.path)

    def addSource(self, sourceDescriptor: SourceDescriptor):
        """Add the given ``sourceDescriptor`` to ``doc.sources``."""
        self.sources.append(sourceDescriptor)

    def addSourceDescriptor(self, **kwargs):
        """Instantiate a new :class:`SourceDescriptor` using the given
        ``kwargs`` and add it to ``doc.sources``.
        """
        source = self.writerClass.sourceDescriptorClass(**kwargs)
        self.addSource(source)
        return source

    def addInstance(self, instanceDescriptor: InstanceDescriptor):
        """Add the given ``instanceDescriptor`` to :attr:`instances`."""
        self.instances.append(instanceDescriptor)

    def addInstanceDescriptor(self, **kwargs):
        """Instantiate a new :class:`InstanceDescriptor` using the given
        ``kwargs`` and add it to :attr:`instances`.
        """
        instance = self.writerClass.instanceDescriptorClass(**kwargs)
        self.addInstance(instance)
        return instance

    def addAxis(self, axisDescriptor: Union[AxisDescriptor, DiscreteAxisDescriptor]):
        """Add the given ``axisDescriptor`` to :attr:`axes`."""
        self.axes.append(axisDescriptor)

    def addAxisDescriptor(self, **kwargs):
        """Instantiate a new :class:`AxisDescriptor` using the given
        ``kwargs`` and add it to :attr:`axes`.

        The axis will be and instance of :class:`DiscreteAxisDescriptor` if
        the ``kwargs`` provide a ``value``, or a :class:`AxisDescriptor` otherwise.
        """
        if 'values' in kwargs:
            axis = self.writerClass.discreteAxisDescriptorClass(**kwargs)
        else:
            axis = self.writerClass.axisDescriptorClass(**kwargs)
        self.addAxis(axis)
        return axis

    def addAxisMapping(self, axisMappingDescriptor: AxisMappingDescriptor):
        """Add the given ``axisMappingDescriptor`` to :attr:`axisMappings`."""
        self.axisMappings.append(axisMappingDescriptor)

    def addAxisMappingDescriptor(self, **kwargs):
        """Instantiate a new :class:`AxisMappingDescriptor` using the given
        ``kwargs`` and add it to :attr:`rules`.
        """
        axisMapping = self.writerClass.axisMappingDescriptorClass(**kwargs)
        self.addAxisMapping(axisMapping)
        return axisMapping

    def addRule(self, ruleDescriptor: RuleDescriptor):
        """Add the given ``ruleDescriptor`` to :attr:`rules`."""
        self.rules.append(ruleDescriptor)

    def addRuleDescriptor(self, **kwargs):
        """Instantiate a new :class:`RuleDescriptor` using the given
        ``kwargs`` and add it to :attr:`rules`.
        """
        rule = self.writerClass.ruleDescriptorClass(**kwargs)
        self.addRule(rule)
        return rule

    def addVariableFont(self, variableFontDescriptor: VariableFontDescriptor):
        """Add the given ``variableFontDescriptor`` to :attr:`variableFonts`.

        .. versionadded:: 5.0
        """
        self.variableFonts.append(variableFontDescriptor)

    def addVariableFontDescriptor(self, **kwargs):
        """Instantiate a new :class:`VariableFontDescriptor` using the given
        ``kwargs`` and add it to :attr:`variableFonts`.

        .. versionadded:: 5.0
        """
        variableFont = self.writerClass.variableFontDescriptorClass(**kwargs)
        self.addVariableFont(variableFont)
        return variableFont

    def addLocationLabel(self, locationLabelDescriptor: LocationLabelDescriptor):
        """Add the given ``locationLabelDescriptor`` to :attr:`locationLabels`.

        .. versionadded:: 5.0
        """
        self.locationLabels.append(locationLabelDescriptor)

    def addLocationLabelDescriptor(self, **kwargs):
        """Instantiate a new :class:`LocationLabelDescriptor` using the given
        ``kwargs`` and add it to :attr:`locationLabels`.

        .. versionadded:: 5.0
        """
        locationLabel = self.writerClass.locationLabelDescriptorClass(**kwargs)
        self.addLocationLabel(locationLabel)
        return locationLabel

    def newDefaultLocation(self):
        """Return a dict with the default location in design space coordinates."""
        loc = collections.OrderedDict()
        for axisDescriptor in self.axes:
            loc[axisDescriptor.name] = axisDescriptor.map_forward(axisDescriptor.default)
        return loc

    def labelForUserLocation(self, userLocation: SimpleLocationDict) -> Optional[LocationLabelDescriptor]:
        """Return the :class:`LocationLabel` that matches the given
        ``userLocation``, or ``None`` if no such label exists.

        .. versionadded:: 5.0
        """
        return next((label for label in self.locationLabels if label.userLocation == userLocation), None)

    def updateFilenameFromPath(self, masters=True, instances=True, force=False):
        """Set a descriptor filename attr from the path and this document path.

        If the filename attribute is not None: skip it.
        """
        if masters:
            for descriptor in self.sources:
                if descriptor.filename is not None and (not force):
                    continue
                if self.path is not None:
                    descriptor.filename = self._posixRelativePath(descriptor.path)
        if instances:
            for descriptor in self.instances:
                if descriptor.filename is not None and (not force):
                    continue
                if self.path is not None:
                    descriptor.filename = self._posixRelativePath(descriptor.path)

    def newAxisDescriptor(self):
        """Ask the writer class to make us a new axisDescriptor."""
        return self.writerClass.getAxisDecriptor()

    def newSourceDescriptor(self):
        """Ask the writer class to make us a new sourceDescriptor."""
        return self.writerClass.getSourceDescriptor()

    def newInstanceDescriptor(self):
        """Ask the writer class to make us a new instanceDescriptor."""
        return self.writerClass.getInstanceDescriptor()

    def getAxisOrder(self):
        """Return a list of axis names, in the same order as defined in the document."""
        names = []
        for axisDescriptor in self.axes:
            names.append(axisDescriptor.name)
        return names

    def getAxis(self, name: str) -> AxisDescriptor | DiscreteAxisDescriptor | None:
        """Return the axis with the given ``name``, or ``None`` if no such axis exists."""
        return next((axis for axis in self.axes if axis.name == name), None)

    def getAxisByTag(self, tag: str) -> AxisDescriptor | DiscreteAxisDescriptor | None:
        """Return the axis with the given ``tag``, or ``None`` if no such axis exists."""
        return next((axis for axis in self.axes if axis.tag == tag), None)

    def getLocationLabel(self, name: str) -> Optional[LocationLabelDescriptor]:
        """Return the top-level location label with the given ``name``, or
        ``None`` if no such label exists.

        .. versionadded:: 5.0
        """
        for label in self.locationLabels:
            if label.name == name:
                return label
        return None

    def map_forward(self, userLocation: SimpleLocationDict) -> SimpleLocationDict:
        """Map a user location to a design location.

        Assume that missing coordinates are at the default location for that axis.

        Note: the output won't be anisotropic, only the xvalue is set.

        .. versionadded:: 5.0
        """
        return {axis.name: axis.map_forward(userLocation.get(axis.name, axis.default)) for axis in self.axes}

    def map_backward(self, designLocation: AnisotropicLocationDict) -> SimpleLocationDict:
        """Map a design location to a user location.

        Assume that missing coordinates are at the default location for that axis.

        When the input has anisotropic locations, only the xvalue is used.

        .. versionadded:: 5.0
        """
        return {axis.name: axis.map_backward(designLocation[axis.name]) if axis.name in designLocation else axis.default for axis in self.axes}

    def findDefault(self):
        """Set and return SourceDescriptor at the default location or None.

        The default location is the set of all `default` values in user space
        of all axes.

        This function updates the document's :attr:`default` value.

        .. versionchanged:: 5.0
           Allow the default source to not specify some of the axis values, and
           they are assumed to be the default.
           See :meth:`SourceDescriptor.getFullDesignLocation()`
        """
        self.default = None
        defaultDesignLocation = self.newDefaultLocation()
        for sourceDescriptor in self.sources:
            if sourceDescriptor.getFullDesignLocation(self) == defaultDesignLocation:
                self.default = sourceDescriptor
                return sourceDescriptor
        return None

    def normalizeLocation(self, location):
        """Return a dict with normalized axis values."""
        from fontTools.varLib.models import normalizeValue
        new = {}
        for axis in self.axes:
            if axis.name not in location:
                continue
            value = location[axis.name]
            if isinstance(value, tuple):
                value = value[0]
            triple = [axis.map_forward(v) for v in (axis.minimum, axis.default, axis.maximum)]
            new[axis.name] = normalizeValue(value, triple)
        return new

    def normalize(self):
        """
        Normalise the geometry of this designspace:

        - scale all the locations of all masters and instances to the -1 - 0 - 1 value.
        - we need the axis data to do the scaling, so we do those last.
        """
        for item in self.sources:
            item.location = self.normalizeLocation(item.location)
        for item in self.instances:
            for _, glyphData in item.glyphs.items():
                glyphData['instanceLocation'] = self.normalizeLocation(glyphData['instanceLocation'])
                for glyphMaster in glyphData['masters']:
                    glyphMaster['location'] = self.normalizeLocation(glyphMaster['location'])
            item.location = self.normalizeLocation(item.location)
        for axis in self.axes:
            newMap = []
            for inputValue, outputValue in axis.map:
                newOutputValue = self.normalizeLocation({axis.name: outputValue}).get(axis.name)
                newMap.append((inputValue, newOutputValue))
            if newMap:
                axis.map = newMap
            minimum = self.normalizeLocation({axis.name: axis.minimum}).get(axis.name)
            maximum = self.normalizeLocation({axis.name: axis.maximum}).get(axis.name)
            default = self.normalizeLocation({axis.name: axis.default}).get(axis.name)
            axis.minimum = minimum
            axis.maximum = maximum
            axis.default = default
        for rule in self.rules:
            newConditionSets = []
            for conditions in rule.conditionSets:
                newConditions = []
                for cond in conditions:
                    if cond.get('minimum') is not None:
                        minimum = self.normalizeLocation({cond['name']: cond['minimum']}).get(cond['name'])
                    else:
                        minimum = None
                    if cond.get('maximum') is not None:
                        maximum = self.normalizeLocation({cond['name']: cond['maximum']}).get(cond['name'])
                    else:
                        maximum = None
                    newConditions.append(dict(name=cond['name'], minimum=minimum, maximum=maximum))
                newConditionSets.append(newConditions)
            rule.conditionSets = newConditionSets

    def loadSourceFonts(self, opener, **kwargs):
        """Ensure SourceDescriptor.font attributes are loaded, and return list of fonts.

        Takes a callable which initializes a new font object (e.g. TTFont, or
        defcon.Font, etc.) from the SourceDescriptor.path, and sets the
        SourceDescriptor.font attribute.
        If the font attribute is already not None, it is not loaded again.
        Fonts with the same path are only loaded once and shared among SourceDescriptors.

        For example, to load UFO sources using defcon:

            designspace = DesignSpaceDocument.fromfile("path/to/my.designspace")
            designspace.loadSourceFonts(defcon.Font)

        Or to load masters as FontTools binary fonts, including extra options:

            designspace.loadSourceFonts(ttLib.TTFont, recalcBBoxes=False)

        Args:
            opener (Callable): takes one required positional argument, the source.path,
                and an optional list of keyword arguments, and returns a new font object
                loaded from the path.
            **kwargs: extra options passed on to the opener function.

        Returns:
            List of font objects in the order they appear in the sources list.
        """
        loaded = {}
        fonts = []
        for source in self.sources:
            if source.font is not None:
                fonts.append(source.font)
                continue
            if source.path in loaded:
                source.font = loaded[source.path]
            else:
                if source.path is None:
                    raise DesignSpaceDocumentError("Designspace source '%s' has no 'path' attribute" % (source.name or '<Unknown>'))
                source.font = opener(source.path, **kwargs)
                loaded[source.path] = source.font
            fonts.append(source.font)
        return fonts

    @property
    def formatTuple(self):
        """Return the formatVersion as a tuple of (major, minor).

        .. versionadded:: 5.0
        """
        if self.formatVersion is None:
            return (5, 0)
        numbers = (int(i) for i in self.formatVersion.split('.'))
        major = next(numbers)
        minor = next(numbers, 0)
        return (major, minor)

    def getVariableFonts(self) -> List[VariableFontDescriptor]:
        """Return all variable fonts defined in this document, or implicit
        variable fonts that can be built from the document's continuous axes.

        In the case of Designspace documents before version 5, the whole
        document was implicitly describing a variable font that covers the
        whole space.

        In version 5 and above documents, there can be as many variable fonts
        as there are locations on discrete axes.

        .. seealso:: :func:`splitInterpolable`

        .. versionadded:: 5.0
        """
        if self.variableFonts:
            return self.variableFonts
        variableFonts = []
        discreteAxes = []
        rangeAxisSubsets: List[Union[RangeAxisSubsetDescriptor, ValueAxisSubsetDescriptor]] = []
        for axis in self.axes:
            if hasattr(axis, 'values'):
                axis = cast(DiscreteAxisDescriptor, axis)
                discreteAxes.append(axis)
            else:
                rangeAxisSubsets.append(RangeAxisSubsetDescriptor(name=axis.name))
        valueCombinations = itertools.product(*[axis.values for axis in discreteAxes])
        for values in valueCombinations:
            basename = None
            if self.filename is not None:
                basename = os.path.splitext(self.filename)[0] + '-VF'
            if self.path is not None:
                basename = os.path.splitext(os.path.basename(self.path))[0] + '-VF'
            if basename is None:
                basename = 'VF'
            axisNames = ''.join([f'-{axis.tag}{value}' for axis, value in zip(discreteAxes, values)])
            variableFonts.append(VariableFontDescriptor(name=f'{basename}{axisNames}', axisSubsets=rangeAxisSubsets + [ValueAxisSubsetDescriptor(name=axis.name, userValue=value) for axis, value in zip(discreteAxes, values)]))
        return variableFonts

    def deepcopyExceptFonts(self):
        """Allow deep-copying a DesignSpace document without deep-copying
        attached UFO fonts or TTFont objects. The :attr:`font` attribute
        is shared by reference between the original and the copy.

        .. versionadded:: 5.0
        """
        fonts = [source.font for source in self.sources]
        try:
            for source in self.sources:
                source.font = None
            res = copy.deepcopy(self)
            for source, font in zip(res.sources, fonts):
                source.font = font
            return res
        finally:
            for source, font in zip(self.sources, fonts):
                source.font = font