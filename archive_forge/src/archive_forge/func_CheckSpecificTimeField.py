from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def CheckSpecificTimeField(unused_instance_ref, args, patch_request):
    """Hook to check specific time field of the request."""
    if args.IsSpecified('reschedule_type'):
        if args.reschedule_type.lower() == 'specific-time':
            if args.IsSpecified('schedule_time'):
                return patch_request
            else:
                raise NoScheduleTimeSpecifiedError('Must specify schedule time')
    return patch_request