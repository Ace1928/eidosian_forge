from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TypeAnnotationValueValuesEnum(_messages.Enum):
    """The TypeAnnotationCode that disambiguates SQL type that Spanner will
    use to represent values of this type during query processing. This is
    necessary for some type codes because a single TypeCode can be mapped to
    different SQL types depending on the SQL dialect. type_annotation
    typically is not needed to process the content of a value (it doesn't
    affect serialization) and clients can ignore it on the read path.

    Values:
      TYPE_ANNOTATION_CODE_UNSPECIFIED: Not specified.
      PG_NUMERIC: PostgreSQL compatible NUMERIC type. This annotation needs to
        be applied to Type instances having NUMERIC type code to specify that
        values of this type should be treated as PostgreSQL NUMERIC values.
        Currently this annotation is always needed for NUMERIC when a client
        interacts with PostgreSQL-enabled Spanner databases.
      PG_JSONB: PostgreSQL compatible JSONB type. This annotation needs to be
        applied to Type instances having JSON type code to specify that values
        of this type should be treated as PostgreSQL JSONB values. Currently
        this annotation is always needed for JSON when a client interacts with
        PostgreSQL-enabled Spanner databases.
    """
    TYPE_ANNOTATION_CODE_UNSPECIFIED = 0
    PG_NUMERIC = 1
    PG_JSONB = 2