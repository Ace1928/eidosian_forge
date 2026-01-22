from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommonFeatureSpec(_messages.Message):
    """CommonFeatureSpec contains Hub-wide configuration information

  Fields:
    anthosobservability: Anthos Observability spec
    appdevexperience: Appdevexperience specific spec.
    clusterupgrade: ClusterUpgrade (fleet-based) feature spec.
    dataplanev2: DataplaneV2 feature spec.
    fleetobservability: FleetObservability feature spec.
    helloworld: Hello World-specific spec.
    multiclusteringress: Multicluster Ingress-specific spec.
    workloadmigration: The specification for WorkloadMigration feature.
  """
    anthosobservability = _messages.MessageField('AnthosObservabilityFeatureSpec', 1)
    appdevexperience = _messages.MessageField('AppDevExperienceFeatureSpec', 2)
    clusterupgrade = _messages.MessageField('ClusterUpgradeFleetSpec', 3)
    dataplanev2 = _messages.MessageField('DataplaneV2FeatureSpec', 4)
    fleetobservability = _messages.MessageField('FleetObservabilityFeatureSpec', 5)
    helloworld = _messages.MessageField('HelloWorldFeatureSpec', 6)
    multiclusteringress = _messages.MessageField('MultiClusterIngressFeatureSpec', 7)
    workloadmigration = _messages.MessageField('WorkloadMigrationFeatureSpec', 8)