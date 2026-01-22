import time
import dateparser
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Callable
class TimeItem:
    secs: float

    @property
    def mins(self):
        return self.secs / 60

    @property
    def hrs(self):
        return self.mins / 60

    @property
    def days(self):
        return self.hrs / 24

    @property
    def wks(self):
        return self.days / 7

    @property
    def mnths(self):
        return self.wks / 4

    @property
    def ablstime(self):
        curr_val = self.secs
        dict_val, str_val = ({}, '')
        for tkey, tnum in TimeValues.items():
            if curr_val >= 1:
                tval = tkey[0] if self.short else tkey
                curr_val, curr_num = divmod(curr_val, tnum)
                if type(curr_num) == float:
                    str_val = f'{curr_num:.1f} {tval} ' + str_val
                else:
                    str_val = f'{curr_num} {tval} ' + str_val
                dict_val[tkey] = curr_num
        str_val = str_val.strip()
        return LazyData(string=str_val, value=dict_val, dtype='time')