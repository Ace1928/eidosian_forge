import logging
from datetime import datetime
from tzlocal import utils
from tzlocal.windows_tz import win_tz
Reload the cached localzone. You need to call this if the timezone has changed.