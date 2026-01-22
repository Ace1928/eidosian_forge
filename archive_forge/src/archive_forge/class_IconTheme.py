import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
class IconTheme(IniFile):
    """Class to parse and validate IconThemes"""

    def __init__(self):
        IniFile.__init__(self)

    def __repr__(self):
        return self.name

    def parse(self, file):
        IniFile.parse(self, file, ['Icon Theme', 'KDE Icon Theme'])
        self.dir = os.path.dirname(file)
        nil, self.name = os.path.split(self.dir)

    def getDir(self):
        return self.dir

    def getName(self):
        return self.get('Name', locale=True)

    def getComment(self):
        return self.get('Comment', locale=True)

    def getInherits(self):
        return self.get('Inherits', list=True)

    def getDirectories(self):
        return self.get('Directories', list=True)

    def getScaledDirectories(self):
        return self.get('ScaledDirectories', list=True)

    def getHidden(self):
        return self.get('Hidden', type='boolean')

    def getExample(self):
        return self.get('Example')

    def getSize(self, directory):
        return self.get('Size', type='integer', group=directory)

    def getContext(self, directory):
        return self.get('Context', group=directory)

    def getType(self, directory):
        value = self.get('Type', group=directory)
        if value:
            return value
        else:
            return 'Threshold'

    def getMaxSize(self, directory):
        value = self.get('MaxSize', type='integer', group=directory)
        if value or value == 0:
            return value
        else:
            return self.getSize(directory)

    def getMinSize(self, directory):
        value = self.get('MinSize', type='integer', group=directory)
        if value or value == 0:
            return value
        else:
            return self.getSize(directory)

    def getThreshold(self, directory):
        value = self.get('Threshold', type='integer', group=directory)
        if value or value == 0:
            return value
        else:
            return 2

    def getScale(self, directory):
        value = self.get('Scale', type='integer', group=directory)
        return value or 1

    def checkExtras(self):
        if self.defaultGroup == 'KDE Icon Theme':
            self.warnings.append('[KDE Icon Theme]-Header is deprecated')
        if self.fileExtension == '.theme':
            pass
        elif self.fileExtension == '.desktop':
            self.warnings.append('.desktop fileExtension is deprecated')
        else:
            self.warnings.append('Unknown File extension')
        try:
            self.name = self.content[self.defaultGroup]['Name']
        except KeyError:
            self.errors.append("Key 'Name' is missing")
        try:
            self.comment = self.content[self.defaultGroup]['Comment']
        except KeyError:
            self.errors.append("Key 'Comment' is missing")
        try:
            self.directories = self.content[self.defaultGroup]['Directories']
        except KeyError:
            self.errors.append("Key 'Directories' is missing")

    def checkGroup(self, group):
        if group == self.defaultGroup:
            try:
                self.name = self.content[group]['Name']
            except KeyError:
                self.errors.append("Key 'Name' in Group '%s' is missing" % group)
            try:
                self.name = self.content[group]['Comment']
            except KeyError:
                self.errors.append("Key 'Comment' in Group '%s' is missing" % group)
        elif group in self.getDirectories():
            try:
                self.type = self.content[group]['Type']
            except KeyError:
                self.type = 'Threshold'
            try:
                self.name = self.content[group]['Size']
            except KeyError:
                self.errors.append("Key 'Size' in Group '%s' is missing" % group)
        elif not (re.match('^\\[X-', group) and is_ascii(group)):
            self.errors.append('Invalid Group name: %s' % group)

    def checkKey(self, key, value, group):
        if group == self.defaultGroup:
            if re.match('^Name' + xdg.Locale.regex + '$', key):
                pass
            elif re.match('^Comment' + xdg.Locale.regex + '$', key):
                pass
            elif key == 'Inherits':
                self.checkValue(key, value, list=True)
            elif key == 'Directories':
                self.checkValue(key, value, list=True)
            elif key == 'ScaledDirectories':
                self.checkValue(key, value, list=True)
            elif key == 'Hidden':
                self.checkValue(key, value, type='boolean')
            elif key == 'Example':
                self.checkValue(key, value)
            elif re.match('^X-[a-zA-Z0-9-]+', key):
                pass
            else:
                self.errors.append('Invalid key: %s' % key)
        elif group in self.getDirectories():
            if key == 'Size':
                self.checkValue(key, value, type='integer')
            elif key == 'Context':
                self.checkValue(key, value)
            elif key == 'Type':
                self.checkValue(key, value)
                if value not in ['Fixed', 'Scalable', 'Threshold']:
                    self.errors.append("Key 'Type' must be one out of 'Fixed','Scalable','Threshold', but is %s" % value)
            elif key == 'MaxSize':
                self.checkValue(key, value, type='integer')
                if self.type != 'Scalable':
                    self.errors.append("Key 'MaxSize' give, but Type is %s" % self.type)
            elif key == 'MinSize':
                self.checkValue(key, value, type='integer')
                if self.type != 'Scalable':
                    self.errors.append("Key 'MinSize' give, but Type is %s" % self.type)
            elif key == 'Threshold':
                self.checkValue(key, value, type='integer')
                if self.type != 'Threshold':
                    self.errors.append("Key 'Threshold' give, but Type is %s" % self.type)
            elif key == 'Scale':
                self.checkValue(key, value, type='integer')
            elif re.match('^X-[a-zA-Z0-9-]+', key):
                pass
            else:
                self.errors.append('Invalid key: %s' % key)