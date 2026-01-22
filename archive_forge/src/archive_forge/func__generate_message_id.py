from typing import AsyncIterator, Dict, Optional, Union
import asyncio
import duet
import google.api_core.exceptions as google_exceptions
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
def _generate_message_id(self) -> str:
    message_id = str(self._next_available_message_id)
    self._next_available_message_id += 1
    return message_id