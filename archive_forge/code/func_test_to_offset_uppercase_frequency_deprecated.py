import re
import pytest
from pandas._libs.tslibs import (
@pytest.mark.parametrize('freq_depr', ['2H', '2BH', '2MIN', '2S', '2Us', '2NS'])
def test_to_offset_uppercase_frequency_deprecated(freq_depr):
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
    f"future version, please use '{freq_depr.lower()[1:]}' instead."
    with pytest.raises(FutureWarning, match=depr_msg):
        to_offset(freq_depr)