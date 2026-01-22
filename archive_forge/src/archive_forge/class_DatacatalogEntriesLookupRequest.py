from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogEntriesLookupRequest(_messages.Message):
    """A DatacatalogEntriesLookupRequest object.

  Fields:
    fullyQualifiedName: [Fully Qualified Name
      (FQN)](https://cloud.google.com//data-catalog/docs/fully-qualified-
      names) of the resource. FQNs take two forms: * For non-regionalized
      resources: `{SYSTEM}:{PROJECT}.{PATH_TO_RESOURCE_SEPARATED_WITH_DOTS}` *
      For regionalized resources: `{SYSTEM}:{PROJECT}.{LOCATION_ID}.{PATH_TO_R
      ESOURCE_SEPARATED_WITH_DOTS}` Example for a DPMS table: `dataproc_metast
      ore:{PROJECT_ID}.{LOCATION_ID}.{INSTANCE_ID}.{DATABASE_ID}.{TABLE_ID}`
    linkedResource: The full name of the Google Cloud Platform resource the
      Data Catalog entry represents. For more information, see [Full Resource
      Name] (https://cloud.google.com/apis/design/resource_names#full_resource
      _name). Full names are case-sensitive. For example: * `//bigquery.google
      apis.com/projects/{PROJECT_ID}/datasets/{DATASET_ID}/tables/{TABLE_ID}`
      * `//pubsub.googleapis.com/projects/{PROJECT_ID}/topics/{TOPIC_ID}`
    location: Location where the lookup should be performed. Required to
      lookup entry that is not a part of `DPMS` or `DATAPLEX`
      `integrated_system` using its `fully_qualified_name`. Ignored in other
      cases.
    project: Project where the lookup should be performed. Required to lookup
      entry that is not a part of `DPMS` or `DATAPLEX` `integrated_system`
      using its `fully_qualified_name`. Ignored in other cases.
    sqlResource: The SQL name of the entry. SQL names are case-sensitive.
      Examples: * `pubsub.topic.{PROJECT_ID}.{TOPIC_ID}` *
      `pubsub.topic.{PROJECT_ID}.`\\``{TOPIC.ID.SEPARATED.WITH.DOTS}`\\` *
      `bigquery.table.{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` *
      `bigquery.dataset.{PROJECT_ID}.{DATASET_ID}` * `datacatalog.entry.{PROJE
      CT_ID}.{LOCATION_ID}.{ENTRY_GROUP_ID}.{ENTRY_ID}` Identifiers (`*_ID`)
      should comply with the [Lexical structure in Standard SQL]
      (https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical).
  """
    fullyQualifiedName = _messages.StringField(1)
    linkedResource = _messages.StringField(2)
    location = _messages.StringField(3)
    project = _messages.StringField(4)
    sqlResource = _messages.StringField(5)