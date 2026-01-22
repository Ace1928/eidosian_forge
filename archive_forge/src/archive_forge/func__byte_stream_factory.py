import io
from fixtures import Fixture
def _byte_stream_factory():
    result = io.BytesIO()
    return (result, result)