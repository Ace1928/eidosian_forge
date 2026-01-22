import abc
import datetime
from typing import Dict, List, Optional, overload, TYPE_CHECKING, Union
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine import calibration
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_processor import AbstractProcessor
from cirq_google.engine.abstract_program import AbstractProgram
def _reservation_to_time_slot(self, reservation: quantum.QuantumReservation) -> quantum.QuantumTimeSlot:
    """Changes a reservation object into a time slot object."""
    return quantum.QuantumTimeSlot(processor_name=self._processor_id, start_time=reservation.start_time, end_time=reservation.end_time, time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION)