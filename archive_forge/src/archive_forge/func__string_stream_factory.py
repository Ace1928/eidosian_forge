import io
from fixtures import Fixture
def _string_stream_factory():
    lower = io.BytesIO()
    upper = io.TextIOWrapper(lower, encoding='utf8')
    upper._CHUNK_SIZE = 1
    return (upper, lower)