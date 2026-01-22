import sys
from math import trunc
from typing import (
class KoreanLocale(Locale):
    names = ['ko', 'ko-kr']
    past = '{0} 전'
    future = '{0} 후'
    timeframes = {'now': '지금', 'second': '1초', 'seconds': '{0}초', 'minute': '1분', 'minutes': '{0}분', 'hour': '한시간', 'hours': '{0}시간', 'day': '하루', 'days': '{0}일', 'week': '1주', 'weeks': '{0}주', 'month': '한달', 'months': '{0}개월', 'year': '1년', 'years': '{0}년'}
    special_dayframes = {-3: '그끄제', -2: '그제', -1: '어제', 1: '내일', 2: '모레', 3: '글피', 4: '그글피'}
    special_yearframes = {-2: '제작년', -1: '작년', 1: '내년', 2: '내후년'}
    month_names = ['', '1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
    month_abbreviations = ['', ' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', '10', '11', '12']
    day_names = ['', '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    day_abbreviations = ['', '월', '화', '수', '목', '금', '토', '일']

    def _ordinal_number(self, n: int) -> str:
        ordinals = ['0', '첫', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열']
        if n < len(ordinals):
            return f'{ordinals[n]}번째'
        return f'{n}번째'

    def _format_relative(self, humanized: str, timeframe: TimeFrameLiteral, delta: Union[float, int]) -> str:
        if timeframe in ('day', 'days'):
            special = self.special_dayframes.get(int(delta))
            if special:
                return special
        elif timeframe in ('year', 'years'):
            special = self.special_yearframes.get(int(delta))
            if special:
                return special
        return super()._format_relative(humanized, timeframe, delta)