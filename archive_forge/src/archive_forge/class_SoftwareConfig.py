from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SoftwareConfig(_messages.Message):
    """Specifies the selection and configuration of software inside the
  environment.

  Enums:
    AirflowExecutorTypeValueValuesEnum: The `airflowExecutorType` specifies
      the [executor](https://airflow.apache.org/code.html?highlight=executor#e
      xecutors) by which task instances are run on Airflow. If this field is
      unspecified, the `airflowExecutorType` defaults to `celery`.
    WebServerPluginsModeValueValuesEnum: Optional. Whether or not the web
      server uses custom plugins. If unspecified, the field defaults to
      `PLUGINS_ENABLED`. This field is supported for Cloud Composer
      environments in versions composer-3.*.*-airflow-*.*.* and newer.

  Messages:
    AirflowConfigOverridesValue: Optional. Apache Airflow configuration
      properties to override. Property keys contain the section and property
      names, separated by a hyphen, for example "core-
      dags_are_paused_at_creation". Section names must not contain hyphens
      ("-"), opening square brackets ("["), or closing square brackets ("]").
      The property name must not be empty and must not contain an equals sign
      ("=") or semicolon (";"). Section and property names must not contain a
      period ("."). Apache Airflow configuration property names must be
      written in [snake_case](https://en.wikipedia.org/wiki/Snake_case).
      Property values can contain any character, and can be written in any
      lower/upper case format. Certain Apache Airflow configuration property
      values are [blocked](/composer/docs/concepts/airflow-configurations),
      and cannot be overridden.
    EnvVariablesValue: Optional. Additional environment variables to provide
      to the Apache Airflow scheduler, worker, and webserver processes.
      Environment variable names must match the regular expression `a-zA-Z_*`.
      They cannot specify Apache Airflow software configuration overrides
      (they cannot match the regular expression
      `AIRFLOW__[A-Z0-9_]+__[A-Z0-9_]+`), and they cannot match any of the
      following reserved names: * `AIRFLOW_HOME` * `C_FORCE_ROOT` *
      `CONTAINER_NAME` * `DAGS_FOLDER` * `GCP_PROJECT` * `GCS_BUCKET` *
      `GKE_CLUSTER_NAME` * `SQL_DATABASE` * `SQL_INSTANCE` * `SQL_PASSWORD` *
      `SQL_PROJECT` * `SQL_REGION` * `SQL_USER`
    PypiPackagesValue: Optional. Custom Python Package Index (PyPI) packages
      to be installed in the environment. Keys refer to the lowercase package
      name such as "numpy" and values are the lowercase extras and version
      specifier such as "==1.12.0", "[devel,gcp_api]", or "[devel]>=1.8.2,
      <1.9.2". To specify a package without pinning it to a version specifier,
      use the empty string as the value.

  Fields:
    airflowConfigOverrides: Optional. Apache Airflow configuration properties
      to override. Property keys contain the section and property names,
      separated by a hyphen, for example "core-dags_are_paused_at_creation".
      Section names must not contain hyphens ("-"), opening square brackets
      ("["), or closing square brackets ("]"). The property name must not be
      empty and must not contain an equals sign ("=") or semicolon (";").
      Section and property names must not contain a period ("."). Apache
      Airflow configuration property names must be written in
      [snake_case](https://en.wikipedia.org/wiki/Snake_case). Property values
      can contain any character, and can be written in any lower/upper case
      format. Certain Apache Airflow configuration property values are
      [blocked](/composer/docs/concepts/airflow-configurations), and cannot be
      overridden.
    airflowExecutorType: The `airflowExecutorType` specifies the [executor](ht
      tps://airflow.apache.org/code.html?highlight=executor#executors) by
      which task instances are run on Airflow. If this field is unspecified,
      the `airflowExecutorType` defaults to `celery`.
    cloudDataLineageIntegration: Optional. The configuration for Cloud Data
      Lineage integration.
    envVariables: Optional. Additional environment variables to provide to the
      Apache Airflow scheduler, worker, and webserver processes. Environment
      variable names must match the regular expression `a-zA-Z_*`. They cannot
      specify Apache Airflow software configuration overrides (they cannot
      match the regular expression `AIRFLOW__[A-Z0-9_]+__[A-Z0-9_]+`), and
      they cannot match any of the following reserved names: * `AIRFLOW_HOME`
      * `C_FORCE_ROOT` * `CONTAINER_NAME` * `DAGS_FOLDER` * `GCP_PROJECT` *
      `GCS_BUCKET` * `GKE_CLUSTER_NAME` * `SQL_DATABASE` * `SQL_INSTANCE` *
      `SQL_PASSWORD` * `SQL_PROJECT` * `SQL_REGION` * `SQL_USER`
    imageVersion: The version of the software running in the environment. This
      encapsulates both the version of Cloud Composer functionality and the
      version of Apache Airflow. It must match the regular expression `compose
      r-([0-9]+(\\.[0-9]+\\.[0-9]+(-preview\\.[0-9]+)?)?|latest)-airflow-([0-
      9]+(\\.[0-9]+(\\.[0-9]+)?)?)`. When used as input, the server also checks
      if the provided version is supported and denies the request for an
      unsupported version. The Cloud Composer portion of the image version is
      a full [semantic version](https://semver.org), or an alias in the form
      of major version number or `latest`. When an alias is provided, the
      server replaces it with the current Cloud Composer version that
      satisfies the alias. The Apache Airflow portion of the image version is
      a full semantic version that points to one of the supported Apache
      Airflow versions, or an alias in the form of only major or major.minor
      versions specified. When an alias is provided, the server replaces it
      with the latest Apache Airflow version that satisfies the alias and is
      supported in the given Cloud Composer version. In all cases, the
      resolved image version is stored in the same field. See also [version
      list](/composer/docs/concepts/versioning/composer-versions) and
      [versioning overview](/composer/docs/concepts/versioning/composer-
      versioning-overview).
    pypiPackages: Optional. Custom Python Package Index (PyPI) packages to be
      installed in the environment. Keys refer to the lowercase package name
      such as "numpy" and values are the lowercase extras and version
      specifier such as "==1.12.0", "[devel,gcp_api]", or "[devel]>=1.8.2,
      <1.9.2". To specify a package without pinning it to a version specifier,
      use the empty string as the value.
    pythonVersion: Optional. The major version of Python used to run the
      Apache Airflow scheduler, worker, and webserver processes. Can be set to
      '2' or '3'. If not specified, the default is '3'. Cannot be updated.
      This field is only supported for Cloud Composer environments in versions
      composer-1.*.*-airflow-*.*.*. Environments in newer versions always use
      Python major version 3.
    schedulerCount: Optional. The number of schedulers for Airflow. This field
      is supported for Cloud Composer environments in versions
      composer-1.*.*-airflow-2.*.*.
    webServerPluginsMode: Optional. Whether or not the web server uses custom
      plugins. If unspecified, the field defaults to `PLUGINS_ENABLED`. This
      field is supported for Cloud Composer environments in versions
      composer-3.*.*-airflow-*.*.* and newer.
  """

    class AirflowExecutorTypeValueValuesEnum(_messages.Enum):
        """The `airflowExecutorType` specifies the [executor](https://airflow.apa
    che.org/code.html?highlight=executor#executors) by which task instances
    are run on Airflow. If this field is unspecified, the
    `airflowExecutorType` defaults to `celery`.

    Values:
      AIRFLOW_EXECUTOR_TYPE_UNSPECIFIED: The Airflow executor type is
        unspecified.
      CELERY: The Celery executor will be used.
      KUBERNETES: The Kubernetes executor will be used.
    """
        AIRFLOW_EXECUTOR_TYPE_UNSPECIFIED = 0
        CELERY = 1
        KUBERNETES = 2

    class WebServerPluginsModeValueValuesEnum(_messages.Enum):
        """Optional. Whether or not the web server uses custom plugins. If
    unspecified, the field defaults to `PLUGINS_ENABLED`. This field is
    supported for Cloud Composer environments in versions
    composer-3.*.*-airflow-*.*.* and newer.

    Values:
      WEB_SERVER_PLUGINS_MODE_UNSPECIFIED: Default mode.
      PLUGINS_DISABLED: Web server plugins are not supported.
      PLUGINS_ENABLED: Web server plugins are supported.
    """
        WEB_SERVER_PLUGINS_MODE_UNSPECIFIED = 0
        PLUGINS_DISABLED = 1
        PLUGINS_ENABLED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AirflowConfigOverridesValue(_messages.Message):
        """Optional. Apache Airflow configuration properties to override.
    Property keys contain the section and property names, separated by a
    hyphen, for example "core-dags_are_paused_at_creation". Section names must
    not contain hyphens ("-"), opening square brackets ("["), or closing
    square brackets ("]"). The property name must not be empty and must not
    contain an equals sign ("=") or semicolon (";"). Section and property
    names must not contain a period ("."). Apache Airflow configuration
    property names must be written in
    [snake_case](https://en.wikipedia.org/wiki/Snake_case). Property values
    can contain any character, and can be written in any lower/upper case
    format. Certain Apache Airflow configuration property values are
    [blocked](/composer/docs/concepts/airflow-configurations), and cannot be
    overridden.

    Messages:
      AdditionalProperty: An additional property for a
        AirflowConfigOverridesValue object.

    Fields:
      additionalProperties: Additional properties of type
        AirflowConfigOverridesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AirflowConfigOverridesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EnvVariablesValue(_messages.Message):
        """Optional. Additional environment variables to provide to the Apache
    Airflow scheduler, worker, and webserver processes. Environment variable
    names must match the regular expression `a-zA-Z_*`. They cannot specify
    Apache Airflow software configuration overrides (they cannot match the
    regular expression `AIRFLOW__[A-Z0-9_]+__[A-Z0-9_]+`), and they cannot
    match any of the following reserved names: * `AIRFLOW_HOME` *
    `C_FORCE_ROOT` * `CONTAINER_NAME` * `DAGS_FOLDER` * `GCP_PROJECT` *
    `GCS_BUCKET` * `GKE_CLUSTER_NAME` * `SQL_DATABASE` * `SQL_INSTANCE` *
    `SQL_PASSWORD` * `SQL_PROJECT` * `SQL_REGION` * `SQL_USER`

    Messages:
      AdditionalProperty: An additional property for a EnvVariablesValue
        object.

    Fields:
      additionalProperties: Additional properties of type EnvVariablesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EnvVariablesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PypiPackagesValue(_messages.Message):
        """Optional. Custom Python Package Index (PyPI) packages to be installed
    in the environment. Keys refer to the lowercase package name such as
    "numpy" and values are the lowercase extras and version specifier such as
    "==1.12.0", "[devel,gcp_api]", or "[devel]>=1.8.2, <1.9.2". To specify a
    package without pinning it to a version specifier, use the empty string as
    the value.

    Messages:
      AdditionalProperty: An additional property for a PypiPackagesValue
        object.

    Fields:
      additionalProperties: Additional properties of type PypiPackagesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PypiPackagesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    airflowConfigOverrides = _messages.MessageField('AirflowConfigOverridesValue', 1)
    airflowExecutorType = _messages.EnumField('AirflowExecutorTypeValueValuesEnum', 2)
    cloudDataLineageIntegration = _messages.MessageField('CloudDataLineageIntegration', 3)
    envVariables = _messages.MessageField('EnvVariablesValue', 4)
    imageVersion = _messages.StringField(5)
    pypiPackages = _messages.MessageField('PypiPackagesValue', 6)
    pythonVersion = _messages.StringField(7)
    schedulerCount = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    webServerPluginsMode = _messages.EnumField('WebServerPluginsModeValueValuesEnum', 9)