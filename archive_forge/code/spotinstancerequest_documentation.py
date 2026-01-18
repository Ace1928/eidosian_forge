from boto.ec2.ec2object import TaggedEC2Object
from boto.ec2.launchspecification import LaunchSpecification


    :ivar id: The ID of the Spot Instance Request.
    :ivar price: The maximum hourly price for any Spot Instance launched to
        fulfill the request.
    :ivar type: The Spot Instance request type.
    :ivar state: The state of the Spot Instance request.
    :ivar fault: The fault codes for the Spot Instance request, if any.
    :ivar valid_from: The start date of the request. If this is a one-time
        request, the request becomes active at this date and time and remains
        active until all instances launch, the request expires, or the request is
        canceled. If the request is persistent, the request becomes active at this
        date and time and remains active until it expires or is canceled.
    :ivar valid_until: The end date of the request. If this is a one-time
        request, the request remains active until all instances launch, the request
        is canceled, or this date is reached. If the request is persistent, it
        remains active until it is canceled or this date is reached.
    :ivar launch_group: The instance launch group. Launch groups are Spot
        Instances that launch together and terminate together.
    :ivar launched_availability_zone: foo
    :ivar product_description: The Availability Zone in which the bid is
        launched.
    :ivar availability_zone_group: The Availability Zone group. If you specify
        the same Availability Zone group for all Spot Instance requests, all Spot
        Instances are launched in the same Availability Zone.
    :ivar create_time: The time stamp when the Spot Instance request was
        created.
    :ivar launch_specification: Additional information for launching instances.
    :ivar instance_id: The instance ID, if an instance has been launched to
        fulfill the Spot Instance request.
    :ivar status: The status code and status message describing the Spot
        Instance request.

    