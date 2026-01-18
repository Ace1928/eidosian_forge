from netaddr import iprange_to_cidrs, IPNetwork, cidr_merge, all_matching_cidrs
def test_whole_network_cidr_merge_v6():
    assert cidr_merge(['::/0', 'fe80::1']) == [IPNetwork('::/0')]
    assert cidr_merge(['::/0', '::']) == [IPNetwork('::/0')]
    assert cidr_merge(['::/0', '::192.0.2.0/124', 'ff00::101']) == [IPNetwork('::/0')]
    assert cidr_merge(['0.0.0.0/0', '0.0.0.0', '::/0', '::']) == [IPNetwork('0.0.0.0/0'), IPNetwork('::/0')]