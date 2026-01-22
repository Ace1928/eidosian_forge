import datetime
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.cloud import quantum
from cirq_google.api import v2
from cirq_google.devices import grid_device
from cirq_google.engine import (
def _cancel_reservation(self, reservation_id: str):
    """Cancel a reservation.

        This will only work for reservations inside the processor's
        schedule freeze window.  If you are not sure whether the reservation
        falls within this window, use remove_reservation
        """
    return self.context.client.cancel_reservation(self.project_id, self.processor_id, reservation_id)