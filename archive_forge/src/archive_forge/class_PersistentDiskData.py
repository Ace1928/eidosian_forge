from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PersistentDiskData(_messages.Message):
    """Persistent Disk service-specific Data. Contains information that may not
  be appropriate for the generic DRZ Augmented View. This currently includes
  LSV Colossus Roots and GCS Buckets.

  Fields:
    cfsRoots: Path to Colossus root for an LSV. NOTE: Unlike `cr_ti_guris` and
      `cr_ti_prefixes`, the field `cfs_roots` below does not need to be a GUri
      or GUri prefix. It can simply be any valid CFS or CFS2 Path. The DRZ KR8
      SIG has more details overall, but generally the `cfs_roots` provided
      here should be scoped to an individual Persistent Disk. An example for a
      PD Disk with a disk ID 3277719120423414466, follows: * `cr_ti_guris`
      could be '/cfs2/pj/pd-cloud-prod' as this is a valid GUri present in the
      DG KB and contains enough information to perform location monitoring and
      scope ownership of the Production Object. * `cfs_roots` would be:
      '/cfs2/pj/pd-cloud-staging/lsv000001234@/
      lsv/projects~773365403387~zones~2700~disks~3277719120423414466 ~bank-
      blue-careful-3526-lsv00054DB1B7254BA3/' as this allows us to enumerate
      the files on CFS2 that belong to an individual Disk.
    gcsBucketNames: The GCS Buckets that back this snapshot or image. This is
      required as `cr_ti_prefixes` and `cr_ti_guris` only accept TI resources.
      This should be the globally unique bucket name.
  """
    cfsRoots = _messages.StringField(1, repeated=True)
    gcsBucketNames = _messages.StringField(2, repeated=True)