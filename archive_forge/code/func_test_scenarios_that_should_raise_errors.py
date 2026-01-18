import os
import pytest
from nltk.twitter import Authenticate
@pytest.mark.parametrize('kwargs', [{'subdir': ''}, {'subdir': None}, {'subdir': '/nosuchdir'}, {}, {'creds_file': 'foobar'}, {'creds_file': 'bad_oauth1-1.txt'}, {'creds_file': 'bad_oauth1-2.txt'}, {'creds_file': 'bad_oauth1-3.txt'}])
def test_scenarios_that_should_raise_errors(self, kwargs, auth):
    """Various scenarios that should raise errors"""
    try:
        auth.load_creds(**kwargs)
    except (OSError, ValueError):
        pass
    except Exception as e:
        pytest.fail('Unexpected exception thrown: %s' % e)
    else:
        pytest.fail('OSError exception not thrown.')