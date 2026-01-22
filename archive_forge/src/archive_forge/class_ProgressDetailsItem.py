from openstack import resource
class ProgressDetailsItem(resource.Resource):
    timestamp = resource.Body('timestamp')
    message = resource.Body('message')
    progress = resource.Body('progress')