import os
import time
from datetime import datetime, timezone
import calendar
def timestampToString(value):
    return asctime(time.gmtime(max(0, value + epoch_diff)))