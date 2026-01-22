import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
class TestUnicodeComments:

    @pytest.mark.skipif(sys.version_info < (2, 7), reason='wide unicode')
    def test_issue_55(self):
        round_trip('        name: TEST\n        description: test using\n        author: Harguroicha\n        sql:\n          command: |-\n            select name from testtbl where no = :no\n\n          ci-test:\n          - :no: 04043709 # 小花\n          - :no: 05161690 # 茶\n          - :no: 05293147 # 〇𤋥川\n          - :no: 05338777 # 〇〇啓\n          - :no: 05273867 # 〇\n          - :no: 05205786 # 〇𤦌\n        ')