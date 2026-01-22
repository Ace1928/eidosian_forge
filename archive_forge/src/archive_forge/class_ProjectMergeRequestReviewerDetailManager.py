from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin
class ProjectMergeRequestReviewerDetailManager(ListMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests/{mr_iid}/reviewers'
    _obj_cls = ProjectMergeRequestReviewerDetail
    _from_parent_attrs = {'project_id': 'project_id', 'mr_iid': 'iid'}