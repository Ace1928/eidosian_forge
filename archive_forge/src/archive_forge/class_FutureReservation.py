from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FutureReservation(_messages.Message):
    """A FutureReservation object.

  Enums:
    PlanningStatusValueValuesEnum: Planning state before being submitted for
      evaluation

  Fields:
    autoCreatedReservationsDeleteTime: Future timestamp when the FR auto-
      created reservations will be deleted by Compute Engine. Format of this
      field must be a valid
      href="https://www.ietf.org/rfc/rfc3339.txt">RFC3339 value.
    autoCreatedReservationsDuration: Specifies the duration of auto-created
      reservations. It represents relative time to future reservation
      start_time when auto-created reservations will be automatically deleted
      by Compute Engine. Duration time unit is represented as a count of
      seconds and fractions of seconds at nanosecond resolution.
    autoDeleteAutoCreatedReservations: Setting for enabling or disabling
      automatic deletion for auto-created reservation. If set to true, auto-
      created reservations will be deleted at Future Reservation's end time
      (default) or at user's defined timestamp if any of the
      [auto_created_reservations_delete_time,
      auto_created_reservations_duration] values is specified. For keeping
      auto-created reservation indefinitely, this value should be set to
      false.
    creationTimestamp: [Output Only] The creation timestamp for this future
      reservation in RFC3339 text format.
    description: An optional description of this resource. Provide this
      property when you create the future reservation.
    id: [Output Only] A unique identifier for this future reservation. The
      server defines this identifier.
    kind: [Output Only] Type of the resource. Always compute#futureReservation
      for future reservations.
    name: The name of the resource, provided by the client when initially
      creating the resource. The resource name must be 1-63 characters long,
      and comply with RFC1035. Specifically, the name must be 1-63 characters
      long and match the regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which
      means the first character must be a lowercase letter, and all following
      characters must be a dash, lowercase letter, or digit, except the last
      character, which cannot be a dash.
    namePrefix: Name prefix for the reservations to be created at the time of
      delivery. The name prefix must comply with RFC1035. Maximum allowed
      length for name prefix is 20. Automatically created reservations name
      format will be -date-####.
    planningStatus: Planning state before being submitted for evaluation
    selfLink: [Output Only] Server-defined fully-qualified URL for this
      resource.
    selfLinkWithId: [Output Only] Server-defined URL for this resource with
      the resource id.
    shareSettings: List of Projects/Folders to share with.
    specificSkuProperties: Future Reservation configuration to indicate
      instance properties and total count.
    status: [Output only] Status of the Future Reservation
    timeWindow: Time window for this Future Reservation.
    zone: [Output Only] URL of the Zone where this future reservation resides.
  """

    class PlanningStatusValueValuesEnum(_messages.Enum):
        """Planning state before being submitted for evaluation

    Values:
      DRAFT: Future Reservation is being drafted.
      PLANNING_STATUS_UNSPECIFIED: <no description>
      SUBMITTED: Future Reservation has been submitted for evaluation by GCP.
    """
        DRAFT = 0
        PLANNING_STATUS_UNSPECIFIED = 1
        SUBMITTED = 2
    autoCreatedReservationsDeleteTime = _messages.StringField(1)
    autoCreatedReservationsDuration = _messages.MessageField('Duration', 2)
    autoDeleteAutoCreatedReservations = _messages.BooleanField(3)
    creationTimestamp = _messages.StringField(4)
    description = _messages.StringField(5)
    id = _messages.IntegerField(6, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(7, default='compute#futureReservation')
    name = _messages.StringField(8)
    namePrefix = _messages.StringField(9)
    planningStatus = _messages.EnumField('PlanningStatusValueValuesEnum', 10)
    selfLink = _messages.StringField(11)
    selfLinkWithId = _messages.StringField(12)
    shareSettings = _messages.MessageField('ShareSettings', 13)
    specificSkuProperties = _messages.MessageField('FutureReservationSpecificSKUProperties', 14)
    status = _messages.MessageField('FutureReservationStatus', 15)
    timeWindow = _messages.MessageField('FutureReservationTimeWindow', 16)
    zone = _messages.StringField(17)