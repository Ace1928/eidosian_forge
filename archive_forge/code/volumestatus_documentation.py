from boto.ec2.instancestatus import Status, Details

    A list object that contains the results of a call to
    DescribeVolumeStatus request.  Each element of the
    list will be an VolumeStatus object.

    :ivar next_token: If the response was truncated by
        the EC2 service, the next_token attribute of the
        object will contain the string that needs to be
        passed in to the next request to retrieve the next
        set of results.
    