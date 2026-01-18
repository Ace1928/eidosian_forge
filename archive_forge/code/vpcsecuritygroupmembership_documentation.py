
    Represents VPC Security Group that this RDS database is a member of

    Properties reference available from the AWS documentation at
    http://docs.aws.amazon.com/AmazonRDS/latest/APIReference/    API_VpcSecurityGroupMembership.html

    Example::
        pri = "sg-abcdefgh"
        sec = "sg-hgfedcba"

        # Create with list of str
        db = c.create_dbinstance(... vpc_security_groups=[pri], ... )

        # Modify with list of str
        db.modify(... vpc_security_groups=[pri,sec], ... )

        # Create with objects
        memberships = []
        membership = VPCSecurityGroupMembership()
        membership.vpc_group = pri
        memberships.append(membership)

        db = c.create_dbinstance(... vpc_security_groups=memberships, ... )

        # Modify with objects
        memberships = d.vpc_security_groups
        membership = VPCSecurityGroupMembership()
        membership.vpc_group = sec
        memberships.append(membership)

        db.modify(...  vpc_security_groups=memberships, ... )

    :ivar connection: :py:class:`boto.rds.RDSConnection` associated with the
        current object
    :ivar vpc_group: This id of the VPC security group
    :ivar status: Status of the VPC security group membership
        <boto.ec2.securitygroup.SecurityGroup>` objects that this RDS Instance
        is a member of
    