import sys
from math import trunc
from typing import (
class SpanishLocale(Locale):
    names = ['es', 'es-es']
    past = 'hace {0}'
    future = 'en {0}'
    and_word = 'y'
    timeframes = {'now': 'ahora', 'second': 'un segundo', 'seconds': '{0} segundos', 'minute': 'un minuto', 'minutes': '{0} minutos', 'hour': 'una hora', 'hours': '{0} horas', 'day': 'un día', 'days': '{0} días', 'week': 'una semana', 'weeks': '{0} semanas', 'month': 'un mes', 'months': '{0} meses', 'year': 'un año', 'years': '{0} años'}
    meridians = {'am': 'am', 'pm': 'pm', 'AM': 'AM', 'PM': 'PM'}
    month_names = ['', 'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    month_abbreviations = ['', 'ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
    day_names = ['', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
    day_abbreviations = ['', 'lun', 'mar', 'mie', 'jue', 'vie', 'sab', 'dom']
    ordinal_day_re = '((?P<value>[1-3]?[0-9](?=[ºª]))[ºª])'

    def _ordinal_number(self, n: int) -> str:
        return f'{n}º'