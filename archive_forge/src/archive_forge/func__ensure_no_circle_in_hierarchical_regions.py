import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def _ensure_no_circle_in_hierarchical_regions(self, region_ref):
    if region_ref.get('parent_region_id') is None:
        return
    root_region_id = region_ref['id']
    parent_region_id = region_ref['parent_region_id']
    while parent_region_id:
        if parent_region_id == root_region_id:
            raise exception.CircularRegionHierarchyError(parent_region_id=parent_region_id)
        parent_region = self.get_region(parent_region_id)
        parent_region_id = parent_region.get('parent_region_id')