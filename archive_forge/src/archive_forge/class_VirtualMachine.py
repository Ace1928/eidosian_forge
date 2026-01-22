from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VirtualMachine(_messages.Message):
    """Carries information about a Compute Engine VM resource.

  Messages:
    LabelsValue: Optional set of labels to apply to the VM and any attached
      disk resources. These labels must adhere to the [name and value
      restrictions](https://cloud.google.com/compute/docs/labeling-resources)
      on VM labels imposed by Compute Engine. Labels keys with the prefix
      'google-' are reserved for use by Google. Labels applied at creation
      time to the VM. Applied on a best-effort basis to attached disk
      resources shortly after VM creation.

  Fields:
    accelerators: The list of accelerators to attach to the VM.
    bootDiskSizeGb: The size of the boot disk, in GB. The boot disk must be
      large enough to accommodate all of the Docker images from each action in
      the pipeline at the same time. If not specified, a small but reasonable
      default value is used.
    bootImage: The host operating system image to use. Currently, only
      Container-Optimized OS images can be used. The default value is
      `projects/cos-cloud/global/images/family/cos-stable`, which selects the
      latest stable release of Container-Optimized OS. This option is provided
      to allow testing against the beta release of the operating system to
      ensure that the new version does not interact negatively with production
      pipelines. To test a pipeline against the beta release of Container-
      Optimized OS, use the value `projects/cos-
      cloud/global/images/family/cos-beta`.
    cpuPlatform: The CPU platform to request. An instance based on a newer
      platform can be allocated, but never one with fewer capabilities. The
      value of this parameter must be a valid Compute Engine CPU platform name
      (such as "Intel Skylake"). This parameter is only useful for carefully
      optimized work loads where the CPU platform has a significant impact.
      For more information about the effect of this parameter, see
      https://cloud.google.com/compute/docs/instances/specify-min-cpu-
      platform.
    disks: The list of disks to create and attach to the VM. Specify either
      the `volumes[]` field or the `disks[]` field, but not both.
    dockerCacheImages: The Compute Engine Disk Images to use as a Docker
      cache. The disks will be mounted into the Docker folder in a way that
      the images present in the cache will not need to be pulled. The digests
      of the cached images must match those of the tags used or the latest
      version will still be pulled. The root directory of the ext4 image must
      contain `image` and `overlay2` directories copied from the Docker
      directory of a VM where the desired Docker images have already been
      pulled. Any images pulled that are not cached will be stored on the
      first cache disk instead of the boot disk. Only a single image is
      supported.
    enableStackdriverMonitoring: Whether Stackdriver monitoring should be
      enabled on the VM.
    labels: Optional set of labels to apply to the VM and any attached disk
      resources. These labels must adhere to the [name and value
      restrictions](https://cloud.google.com/compute/docs/labeling-resources)
      on VM labels imposed by Compute Engine. Labels keys with the prefix
      'google-' are reserved for use by Google. Labels applied at creation
      time to the VM. Applied on a best-effort basis to attached disk
      resources shortly after VM creation.
    machineType: Required. The machine type of the virtual machine to create.
      Must be the short name of a standard machine type (such as
      "n1-standard-1") or a custom machine type (such as "custom-1-4096",
      where "1" indicates the number of vCPUs and "4096" indicates the memory
      in MB). See [Creating an instance with a custom machine
      type](https://cloud.google.com/compute/docs/instances/creating-instance-
      with-custom-machine-type#create) for more specifications on creating a
      custom machine type.
    network: The VM network configuration.
    nvidiaDriverVersion: The NVIDIA driver version to use when attaching an
      NVIDIA GPU accelerator. The version specified here must be compatible
      with the GPU libraries contained in the container being executed, and
      must be one of the drivers hosted in the `nvidia-drivers-us-public`
      bucket on Google Cloud Storage.
    preemptible: If true, allocate a preemptible VM.
    reservation: If specified, the VM will only be allocated inside the
      matching reservation. It will fail if the VM parameters don't match the
      reservation.
    serviceAccount: The service account to install on the VM. This account
      does not need any permissions other than those required by the pipeline.
    volumes: The list of disks and other storage to create or attach to the
      VM. Specify either the `volumes[]` field or the `disks[]` field, but not
      both.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional set of labels to apply to the VM and any attached disk
    resources. These labels must adhere to the [name and value
    restrictions](https://cloud.google.com/compute/docs/labeling-resources) on
    VM labels imposed by Compute Engine. Labels keys with the prefix 'google-'
    are reserved for use by Google. Labels applied at creation time to the VM.
    Applied on a best-effort basis to attached disk resources shortly after VM
    creation.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    accelerators = _messages.MessageField('Accelerator', 1, repeated=True)
    bootDiskSizeGb = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    bootImage = _messages.StringField(3)
    cpuPlatform = _messages.StringField(4)
    disks = _messages.MessageField('Disk', 5, repeated=True)
    dockerCacheImages = _messages.StringField(6, repeated=True)
    enableStackdriverMonitoring = _messages.BooleanField(7)
    labels = _messages.MessageField('LabelsValue', 8)
    machineType = _messages.StringField(9)
    network = _messages.MessageField('Network', 10)
    nvidiaDriverVersion = _messages.StringField(11)
    preemptible = _messages.BooleanField(12)
    reservation = _messages.StringField(13)
    serviceAccount = _messages.MessageField('ServiceAccount', 14)
    volumes = _messages.MessageField('Volume', 15, repeated=True)