import sys
from math import trunc
from typing import (
class ItalianLocale(Locale):
    names = ['it', 'it-it']
    past = '{0} fa'
    future = 'tra {0}'
    and_word = 'e'
    timeframes = {'now': 'adesso', 'second': 'un secondo', 'seconds': '{0} qualche secondo', 'minute': 'un minuto', 'minutes': '{0} minuti', 'hour': "un'ora", 'hours': '{0} ore', 'day': 'un giorno', 'days': '{0} giorni', 'week': 'una settimana', 'weeks': '{0} settimane', 'month': 'un mese', 'months': '{0} mesi', 'year': 'un anno', 'years': '{0} anni'}
    month_names = ['', 'gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno', 'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre']
    month_abbreviations = ['', 'gen', 'feb', 'mar', 'apr', 'mag', 'giu', 'lug', 'ago', 'set', 'ott', 'nov', 'dic']
    day_names = ['', 'lunedì', 'martedì', 'mercoledì', 'giovedì', 'venerdì', 'sabato', 'domenica']
    day_abbreviations = ['', 'lun', 'mar', 'mer', 'gio', 'ven', 'sab', 'dom']
    ordinal_day_re = '((?P<value>[1-3]?[0-9](?=[ºª]))[ºª])'

    def _ordinal_number(self, n: int) -> str:
        return f'{n}º'