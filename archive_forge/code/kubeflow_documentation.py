import logging
import os
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
Environment for distributed training using the `PyTorchJob`_ operator from `Kubeflow`_.

    This environment, unlike others, does not get auto-detected and needs to be passed to the Fabric/Trainer
    constructor manually.

    .. _PyTorchJob: https://www.kubeflow.org/docs/components/training/pytorch/
    .. _Kubeflow: https://www.kubeflow.org

    