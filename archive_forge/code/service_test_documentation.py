from typing import Optional, Iterator
import pytest
import httpx
from cirq_rigetti import get_rigetti_qcs_service, RigettiQCSService
test that `RigettiQCSService` will use a custom defined client when the
    user specifies one to make an API call.