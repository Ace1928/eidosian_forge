import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
def DirectorySizeDistance(subdir, iconsize, theme):
    Type = theme.getType(subdir)
    Size = theme.getSize(subdir)
    Threshold = theme.getThreshold(subdir)
    MinSize = theme.getMinSize(subdir)
    MaxSize = theme.getMaxSize(subdir)
    if Type == 'Fixed':
        return abs(Size - iconsize)
    elif Type == 'Scalable':
        if iconsize < MinSize:
            return MinSize - iconsize
        elif iconsize > MaxSize:
            return MaxSize - iconsize
        return 0
    elif Type == 'Threshold':
        if iconsize < Size - Threshold:
            return MinSize - iconsize
        elif iconsize > Size + Threshold:
            return iconsize - MaxSize
        return 0