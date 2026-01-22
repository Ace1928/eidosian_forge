import collections
import enum
from hyperframe.frame import SettingsFrame
from h2.errors import ErrorCodes
from h2.exceptions import InvalidSettingsValueError
class ChangedSetting:

    def __init__(self, setting, original_value, new_value):
        self.setting = setting
        self.original_value = original_value
        self.new_value = new_value

    def __repr__(self):
        return 'ChangedSetting(setting=%s, original_value=%s, new_value=%s)' % (self.setting, self.original_value, self.new_value)