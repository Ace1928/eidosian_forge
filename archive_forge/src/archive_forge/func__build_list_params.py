def _build_list_params(self, params, prefix=''):
    i = 1
    for dev_name in self:
        pre = '%s.%d' % (prefix, i)
        params['%s.DeviceName' % pre] = dev_name
        block_dev = self[dev_name]
        if block_dev.ephemeral_name:
            params['%s.VirtualName' % pre] = block_dev.ephemeral_name
        elif block_dev.no_device:
            params['%s.NoDevice' % pre] = ''
        else:
            if block_dev.snapshot_id:
                params['%s.Ebs.SnapshotId' % pre] = block_dev.snapshot_id
            if block_dev.size:
                params['%s.Ebs.VolumeSize' % pre] = block_dev.size
            if block_dev.delete_on_termination:
                params['%s.Ebs.DeleteOnTermination' % pre] = 'true'
            else:
                params['%s.Ebs.DeleteOnTermination' % pre] = 'false'
            if block_dev.volume_type:
                params['%s.Ebs.VolumeType' % pre] = block_dev.volume_type
            if block_dev.iops is not None:
                params['%s.Ebs.Iops' % pre] = block_dev.iops
            if block_dev.encrypted is not None:
                if block_dev.encrypted:
                    params['%s.Ebs.Encrypted' % pre] = 'true'
                else:
                    params['%s.Ebs.Encrypted' % pre] = 'false'
        i += 1