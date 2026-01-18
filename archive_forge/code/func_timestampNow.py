import os
import time
from datetime import datetime, timezone
import calendar
def timestampNow():
    source_date_epoch = os.environ.get('SOURCE_DATE_EPOCH')
    if source_date_epoch is not None:
        return int(source_date_epoch) - epoch_diff
    return int(time.time() - epoch_diff)