from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1MigrationProgressEvent(_messages.Message):
    """An event signifying the start of a new step in a [migration from Cloud
  Datastore to Cloud Firestore in Datastore
  mode](https://cloud.google.com/datastore/docs/upgrade-to-firestore).

  Enums:
    StepValueValuesEnum: The step that is starting. An event with step set to
      `START` indicates that the migration has been reverted back to the
      initial pre-migration state.

  Fields:
    prepareStepDetails: Details for the `PREPARE` step.
    redirectWritesStepDetails: Details for the `REDIRECT_WRITES` step.
    step: The step that is starting. An event with step set to `START`
      indicates that the migration has been reverted back to the initial pre-
      migration state.
  """

    class StepValueValuesEnum(_messages.Enum):
        """The step that is starting. An event with step set to `START` indicates
    that the migration has been reverted back to the initial pre-migration
    state.

    Values:
      MIGRATION_STEP_UNSPECIFIED: Unspecified.
      PREPARE: Pre-migration: the database is prepared for migration.
      START: Start of migration.
      APPLY_WRITES_SYNCHRONOUSLY: Writes are applied synchronously to at least
        one replica.
      COPY_AND_VERIFY: Data is copied to Cloud Firestore and then verified to
        match the data in Cloud Datastore.
      REDIRECT_EVENTUALLY_CONSISTENT_READS: Eventually-consistent reads are
        redirected to Cloud Firestore.
      REDIRECT_STRONGLY_CONSISTENT_READS: Strongly-consistent reads are
        redirected to Cloud Firestore.
      REDIRECT_WRITES: Writes are redirected to Cloud Firestore.
    """
        MIGRATION_STEP_UNSPECIFIED = 0
        PREPARE = 1
        START = 2
        APPLY_WRITES_SYNCHRONOUSLY = 3
        COPY_AND_VERIFY = 4
        REDIRECT_EVENTUALLY_CONSISTENT_READS = 5
        REDIRECT_STRONGLY_CONSISTENT_READS = 6
        REDIRECT_WRITES = 7
    prepareStepDetails = _messages.MessageField('GoogleDatastoreAdminV1PrepareStepDetails', 1)
    redirectWritesStepDetails = _messages.MessageField('GoogleDatastoreAdminV1RedirectWritesStepDetails', 2)
    step = _messages.EnumField('StepValueValuesEnum', 3)