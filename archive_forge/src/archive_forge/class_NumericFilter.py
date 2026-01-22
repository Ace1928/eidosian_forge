from typing import List, Optional, Union
class NumericFilter(Filter):
    INF = '+inf'
    NEG_INF = '-inf'

    def __init__(self, field: str, minval: Union[int, str], maxval: Union[int, str], minExclusive: bool=False, maxExclusive: bool=False) -> None:
        args = [minval if not minExclusive else f'({minval}', maxval if not maxExclusive else f'({maxval}']
        Filter.__init__(self, 'FILTER', field, *args)