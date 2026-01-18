import json
import urllib
import numpy as np
import pytest
import cirq
from cirq import quirk_url_to_circuit, quirk_json_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
@pytest.mark.parametrize('url,error_cls,msg', [('http://algassert.com/quirk#bad', ValueError, 'must start with "circuit="'), ('http://algassert.com/quirk#circuit=', json.JSONDecodeError, None)])
def test_parse_url_failures(url, error_cls, msg):
    with pytest.raises(error_cls, match=msg):
        _ = quirk_url_to_circuit(url)