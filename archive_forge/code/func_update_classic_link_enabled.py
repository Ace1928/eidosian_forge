from boto.ec2.ec2object import TaggedEC2Object
def update_classic_link_enabled(self, validate=False, dry_run=False):
    """
        Updates instance's classic_link_enabled attribute

        :rtype: bool
        :return: self.classic_link_enabled after update has occurred.
        """
    self._get_status_then_update_vpc(self.connection.get_all_classic_link_vpcs, validate=validate, dry_run=dry_run)
    return self.classic_link_enabled