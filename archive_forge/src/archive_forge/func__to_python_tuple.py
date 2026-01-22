import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def _to_python_tuple(self, value, state):
    time = value.strip()
    explicit_ampm = False
    if self.use_ampm:
        last_two = time[-2:].lower()
        if last_two not in ('am', 'pm'):
            if self.use_ampm != 'optional':
                raise Invalid(self.message('noAMPM', state), value, state)
            offset = 0
        else:
            explicit_ampm = True
            offset = 12 if last_two == 'pm' else 0
            time = time[:-2]
    else:
        offset = 0
    parts = time.split(':', 3)
    if len(parts) > 3:
        raise Invalid(self.message('tooManyColon', state), value, state)
    if len(parts) == 3 and (not self.use_seconds):
        raise Invalid(self.message('noSeconds', state), value, state)
    if len(parts) == 2 and self.use_seconds and (self.use_seconds != 'optional'):
        raise Invalid(self.message('secondsRequired', state), value, state)
    if len(parts) == 1:
        raise Invalid(self.message('minutesRequired', state), value, state)
    try:
        hour = int(parts[0])
    except ValueError:
        raise Invalid(self.message('badNumber', state, number=parts[0], part='hour'), value, state)
    if explicit_ampm:
        if not 1 <= hour <= 12:
            raise Invalid(self.message('badHour', state, number=hour, range='1-12'), value, state)
        if hour == 12 and offset == 12:
            pass
        elif hour == 12 and offset == 0:
            hour = 0
        else:
            hour += offset
    elif not 0 <= hour < 24:
        raise Invalid(self.message('badHour', state, number=hour, range='0-23'), value, state)
    try:
        minute = int(parts[1])
    except ValueError:
        raise Invalid(self.message('badNumber', state, number=parts[1], part='minute'), value, state)
    if not 0 <= minute < 60:
        raise Invalid(self.message('badMinute', state, number=minute), value, state)
    if len(parts) == 3:
        try:
            second = int(parts[2])
        except ValueError:
            raise Invalid(self.message('badNumber', state, number=parts[2], part='second'), value, state)
        if not 0 <= second < 60:
            raise Invalid(self.message('badSecond', state, number=second), value, state)
    else:
        second = None
    if second is None:
        return (hour, minute)
    else:
        return (hour, minute, second)