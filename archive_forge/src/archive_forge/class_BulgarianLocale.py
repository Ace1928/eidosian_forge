import sys
from math import trunc
from typing import (
class BulgarianLocale(SlavicBaseLocale):
    names = ['bg', 'bg-bg']
    past = '{0} назад'
    future = 'напред {0}'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'сега', 'second': 'секунда', 'seconds': '{0} няколко секунди', 'minute': 'минута', 'minutes': {'singular': '{0} минута', 'dual': '{0} минути', 'plural': '{0} минути'}, 'hour': 'час', 'hours': {'singular': '{0} час', 'dual': '{0} часа', 'plural': '{0} часа'}, 'day': 'ден', 'days': {'singular': '{0} ден', 'dual': '{0} дни', 'plural': '{0} дни'}, 'month': 'месец', 'months': {'singular': '{0} месец', 'dual': '{0} месеца', 'plural': '{0} месеца'}, 'year': 'година', 'years': {'singular': '{0} година', 'dual': '{0} години', 'plural': '{0} години'}}
    month_names = ['', 'януари', 'февруари', 'март', 'април', 'май', 'юни', 'юли', 'август', 'септември', 'октомври', 'ноември', 'декември']
    month_abbreviations = ['', 'ян', 'февр', 'март', 'апр', 'май', 'юни', 'юли', 'авг', 'септ', 'окт', 'ноем', 'дек']
    day_names = ['', 'понеделник', 'вторник', 'сряда', 'четвъртък', 'петък', 'събота', 'неделя']
    day_abbreviations = ['', 'пон', 'вт', 'ср', 'четв', 'пет', 'съб', 'нед']