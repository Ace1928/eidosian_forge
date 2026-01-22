import abc
from taskflow import exceptions as exc
from taskflow.persistence import base
from taskflow.persistence import models
def _do_update_flow_details(self, flow_detail, transaction, ignore_missing=False):
    flow_path = self._get_obj_path(flow_detail)
    self._update_object(flow_detail, transaction, ignore_missing=ignore_missing)
    for atom_details in flow_detail:
        atom_path = self._get_obj_path(atom_details)
        link_path = self._join_path(flow_path, atom_details.uuid)
        self._create_link(atom_path, link_path, transaction)
        self._update_object(atom_details, transaction, ignore_missing=True)
    return flow_detail