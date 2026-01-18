import datetime
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
def test_reservations():
    p = NothingProcessor(processor_id='test')
    start_reservation = datetime.datetime.now()
    end_reservation = datetime.datetime.now() + datetime.timedelta(hours=2)
    users = ['gooduser@test.com']
    reservation = p.create_reservation(start_time=start_reservation, end_time=end_reservation, whitelisted_users=users)
    assert reservation.start_time.timestamp() == int(start_reservation.timestamp())
    assert reservation.end_time.timestamp() == int(end_reservation.timestamp())
    assert reservation.whitelisted_users == users
    assert p.get_reservation(reservation.name) == reservation
    assert p.get_reservation('nothing_to_see_here') is None
    end_reservation = datetime.datetime.now() + datetime.timedelta(hours=3)
    p.update_reservation(reservation_id=reservation.name, end_time=end_reservation)
    reservation = p.get_reservation(reservation.name)
    assert reservation.end_time.timestamp() == int(end_reservation.timestamp())
    start_reservation = datetime.datetime.now() + datetime.timedelta(hours=1)
    p.update_reservation(reservation_id=reservation.name, start_time=start_reservation)
    reservation = p.get_reservation(reservation.name)
    assert reservation.start_time.timestamp() == int(start_reservation.timestamp())
    users = ['gooduser@test.com', 'otheruser@prod.com']
    p.update_reservation(reservation_id=reservation.name, whitelisted_users=users)
    reservation = p.get_reservation(reservation.name)
    assert reservation.whitelisted_users == users
    with pytest.raises(ValueError, match='does not exist'):
        p.update_reservation(reservation_id='invalid', whitelisted_users=users)