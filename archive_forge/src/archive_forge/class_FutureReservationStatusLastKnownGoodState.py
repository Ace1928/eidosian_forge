from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FutureReservationStatusLastKnownGoodState(_messages.Message):
    """The state that the future reservation will be reverted to should the
  amendment be declined.

  Enums:
    ProcurementStatusValueValuesEnum: [Output Only] The status of the last
      known good state for the Future Reservation.

  Fields:
    description: [Output Only] The description of the FutureReservation before
      an amendment was requested.
    futureReservationSpecs: A
      FutureReservationStatusLastKnownGoodStateFutureReservationSpecs
      attribute.
    lockTime: [Output Only] The lock time of the FutureReservation before an
      amendment was requested.
    namePrefix: [Output Only] The name prefix of the Future Reservation before
      an amendment was requested.
    procurementStatus: [Output Only] The status of the last known good state
      for the Future Reservation.
  """

    class ProcurementStatusValueValuesEnum(_messages.Enum):
        """[Output Only] The status of the last known good state for the Future
    Reservation.

    Values:
      APPROVED: Future reservation is approved by GCP.
      CANCELLED: Future reservation is cancelled by the customer.
      COMMITTED: Future reservation is committed by the customer.
      DECLINED: Future reservation is rejected by GCP.
      DRAFTING: Related status for PlanningStatus.Draft. Transitions to
        PENDING_APPROVAL upon user submitting FR.
      FAILED: Future reservation failed. No additional reservations were
        provided.
      FAILED_PARTIALLY_FULFILLED: Future reservation is partially fulfilled.
        Additional reservations were provided but did not reach total_count
        reserved instance slots.
      FULFILLED: Future reservation is fulfilled completely.
      PENDING_AMENDMENT_APPROVAL: An Amendment to the Future Reservation has
        been requested. If the Amendment is declined, the Future Reservation
        will be restored to the last known good state.
      PENDING_APPROVAL: Future reservation is pending approval by GCP.
      PROCUREMENT_STATUS_UNSPECIFIED: <no description>
      PROCURING: Future reservation is being procured by GCP. Beyond this
        point, Future reservation is locked and no further modifications are
        allowed.
      PROVISIONING: Future reservation capacity is being provisioned. This
        state will be entered after start_time, while reservations are being
        created to provide total_count reserved instance slots. This state
        will not persist past start_time + 24h.
    """
        APPROVED = 0
        CANCELLED = 1
        COMMITTED = 2
        DECLINED = 3
        DRAFTING = 4
        FAILED = 5
        FAILED_PARTIALLY_FULFILLED = 6
        FULFILLED = 7
        PENDING_AMENDMENT_APPROVAL = 8
        PENDING_APPROVAL = 9
        PROCUREMENT_STATUS_UNSPECIFIED = 10
        PROCURING = 11
        PROVISIONING = 12
    description = _messages.StringField(1)
    futureReservationSpecs = _messages.MessageField('FutureReservationStatusLastKnownGoodStateFutureReservationSpecs', 2)
    lockTime = _messages.StringField(3)
    namePrefix = _messages.StringField(4)
    procurementStatus = _messages.EnumField('ProcurementStatusValueValuesEnum', 5)