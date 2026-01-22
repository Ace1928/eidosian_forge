from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricDescriptor(_messages.Message):
    """Defines a metric type and its schema.

  Enums:
    MetricKindValueValuesEnum: Whether the metric records instantaneous
      values, changes to a value, etc.
    ValueTypeValueValuesEnum: Whether the measurement is an integer, a
      floating-point number, etc.

  Fields:
    description: A detailed description of the metric, which can be used in
      documentation.
    displayName: A concise name for the metric, which can be displayed in user
      interfaces. Use sentence case without an ending period, for example
      "Request count".
    labels: The set of labels that can be used to describe a specific instance
      of this metric type. For example, the
      `compute.googleapis.com/instance/network/received_bytes_count` metric
      type has a label, `loadbalanced`, that specifies whether the traffic was
      received through a load balanced IP address.
    metricKind: Whether the metric records instantaneous values, changes to a
      value, etc.
    name: Resource name. The format of the name may vary between different
      implementations. For examples:
      projects/{project_id}/metricDescriptors/{type=**}
      metricDescriptors/{type=**}
    type: The metric type including a DNS name prefix, for example
      `"compute.googleapis.com/instance/cpu/utilization"`. Metric types should
      use a natural hierarchical grouping such as the following:
      compute.googleapis.com/instance/cpu/utilization
      compute.googleapis.com/instance/disk/read_ops_count
      compute.googleapis.com/instance/network/received_bytes_count  Note that
      if the metric type changes, the monitoring data will be discontinued,
      and anything depends on it will break, such as monitoring dashboards,
      alerting rules and quota limits. Therefore, once a metric has been
      published, its type should be immutable.
    unit: The unit in which the metric value is reported. It is only
      applicable if the `value_type` is `INT64`, `DOUBLE`, or `DISTRIBUTION`.
      The supported units are a subset of [The Unified Code for Units of
      Measure](http://unitsofmeasure.org/ucum.html) standard:  **Basic units
      (UNIT)**  * `bit`   bit * `By`    byte * `s`     second * `min`   minute
      * `h`     hour * `d`     day  **Prefixes (PREFIX)**  * `k`     kilo
      (10**3) * `M`     mega    (10**6) * `G`     giga    (10**9) * `T`
      tera    (10**12) * `P`     peta    (10**15) * `E`     exa     (10**18) *
      `Z`     zetta   (10**21) * `Y`     yotta   (10**24) * `m`     milli
      (10**-3) * `u`     micro   (10**-6) * `n`     nano    (10**-9) * `p`
      pico    (10**-12) * `f`     femto   (10**-15) * `a`     atto
      (10**-18) * `z`     zepto   (10**-21) * `y`     yocto   (10**-24) * `Ki`
      kibi    (2**10) * `Mi`    mebi    (2**20) * `Gi`    gibi    (2**30) *
      `Ti`    tebi    (2**40)  **Grammar**  The grammar includes the
      dimensionless unit `1`, such as `1/s`.  The grammar also includes these
      connectors:  * `/`    division (as an infix operator, e.g. `1/s`). * `.`
      multiplication (as an infix operator, e.g. `GBy.d`)  The grammar for a
      unit is as follows:      Expression = Component { "." Component } { "/"
      Component } ;      Component = [ PREFIX ] UNIT [ Annotation ]
      | Annotation               | "1"               ;      Annotation = "{"
      NAME "}" ;  Notes:  * `Annotation` is just a comment if it follows a
      `UNIT` and is    equivalent to `1` if it is used alone. For examples,
      `{requests}/s == 1/s`, `By{transmitted}/s == By/s`. * `NAME` is a
      sequence of non-blank printable ASCII characters not    containing '{'
      or '}'.
    valueType: Whether the measurement is an integer, a floating-point number,
      etc.
  """

    class MetricKindValueValuesEnum(_messages.Enum):
        """Whether the metric records instantaneous values, changes to a value,
    etc.

    Values:
      METRIC_KIND_UNSPECIFIED: Do not use this default value.
      GAUGE: Instantaneous measurements of a varying quantity.
      DELTA: Changes over non-overlapping time intervals.
      CUMULATIVE: Cumulative value over time intervals that can overlap. The
        overlapping intervals must have the same start time.
    """
        METRIC_KIND_UNSPECIFIED = 0
        GAUGE = 1
        DELTA = 2
        CUMULATIVE = 3

    class ValueTypeValueValuesEnum(_messages.Enum):
        """Whether the measurement is an integer, a floating-point number, etc.

    Values:
      VALUE_TYPE_UNSPECIFIED: Do not use this default value.
      BOOL: The value is a boolean. This value type can be used only if the
        metric kind is `GAUGE`.
      INT64: The value is a signed 64-bit integer.
      DOUBLE: The value is a double precision floating point number.
      STRING: The value is a text string. This value type can be used only if
        the metric kind is `GAUGE`.
      DISTRIBUTION: The value is a `Distribution`.
      MONEY: The value is money.
    """
        VALUE_TYPE_UNSPECIFIED = 0
        BOOL = 1
        INT64 = 2
        DOUBLE = 3
        STRING = 4
        DISTRIBUTION = 5
        MONEY = 6
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    labels = _messages.MessageField('LabelDescriptor', 3, repeated=True)
    metricKind = _messages.EnumField('MetricKindValueValuesEnum', 4)
    name = _messages.StringField(5)
    type = _messages.StringField(6)
    unit = _messages.StringField(7)
    valueType = _messages.EnumField('ValueTypeValueValuesEnum', 8)