from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class TcpProfilesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'defaultsFrom': 'parent', 'ackOnPush': 'ack_on_push', 'autoProxyBufferSize': 'auto_proxy_buffer', 'autoReceiveWindowSize': 'auto_receive_window', 'autoSendBufferSize': 'auto_send_buffer', 'closeWaitTimeout': 'close_wait', 'cmetricsCache': 'congestion_metrics_cache', 'cmetricsCacheTimeout': 'congestion_metrics_cache_timeout', 'congestionControl': 'congestion_control', 'deferredAccept': 'deferred_accept', 'delayWindowControl': 'delay_window_control', 'delayedAcks': 'delayed_acks', 'earlyRetransmit': 'early_retransmit', 'ecn': 'explicit_congestion_notification', 'enhancedLossRecovery': 'enhanced_loss_recovery', 'fastOpen': 'fast_open', 'fastOpenCookieExpiration': 'fast_open_cookie_expiration', 'finWaitTimeout': 'fin_wait_1', 'finWait_2Timeout': 'fin_wait_2', 'idleTimeout': 'idle_timeout', 'initCwnd': 'initial_congestion_window_size', 'initRwnd': 'initial_receive_window_size', 'ipDfMode': 'dont_fragment_flag', 'ipTosToClient': 'ip_tos', 'ipTtlMode': 'time_to_live', 'ipTtlV4': 'time_to_live_v4', 'ipTtlV6': 'time_to_live_v6', 'keepAliveInterval': 'keep_alive_interval', 'limitedTransmit': 'limited_transmit_recovery', 'linkQosToClient': 'link_qos', 'maxRetrans': 'max_segment_retrans', 'synMaxRetrans': 'max_syn_retrans', 'rexmtThresh': 'retransmit_threshold', 'maxSegmentSize': 'max_segment_size', 'md5Signature': 'md5_signature', 'minimumRto': 'minimum_rto', 'mptcp': 'multipath_tcp', 'mptcpCsum': 'mptcp_checksum', 'mptcpCsumVerify': 'mptcp_checksum_verify', 'mptcpFallback': 'mptcp_fallback', 'mptcpFastjoin': 'mptcp_fast_join', 'mptcpIdleTimeout': 'mptcp_idle_timeout', 'mptcpJoinMax': 'mptcp_join_max', 'mptcpMakeafterbreak': 'mptcp_make_after_break', 'mptcpNojoindssack': 'mptcp_no_join_dss_ack', 'mptcpRtomax': 'mptcp_rto_max', 'mptcpRxmitmin': 'mptcp_retransmit_min', 'mptcpSubflowmax': 'mptcp_subflow_max', 'mptcpTimeout': 'mptcp_timeout', 'nagle': 'nagle_algorithm', 'pktLossIgnoreBurst': 'pkt_loss_ignore_burst', 'pktLossIgnoreRate': 'pkt_loss_ignore_rate', 'proxyBufferHigh': 'proxy_buffer_high', 'proxyBufferLow': 'proxy_buffer_low', 'proxyMss': 'proxy_max_segment', 'proxyOptions': 'proxy_options', 'pushFlag': 'push_flag', 'ratePace': 'rate_pace', 'ratePaceMaxRate': 'rate_pace_max_rate', 'receiveWindowSize': 'receive_window', 'resetOnTimeout': 'reset_on_timeout', 'selectiveAcks': 'selective_acks', 'selectiveNack': 'selective_nack', 'sendBufferSize': 'send_buffer', 'slowStart': 'slow_start', 'synCookieEnable': 'syn_cookie_enable', 'synCookieWhitelist': 'syn_cookie_white_list', 'synRtoBase': 'syn_retrans_to_base', 'tailLossProbe': 'tail_loss_probe', 'timeWaitRecycle': 'time_wait_recycle', 'timeWaitTimeout': 'time_wait', 'verifiedAccept': 'verified_accept', 'zeroWindowTimeout': 'zero_window_timeout'}
    returnables = ['full_path', 'name', 'parent', 'description', 'abc', 'ack_on_push', 'auto_proxy_buffer', 'auto_receive_window', 'auto_send_buffer', 'close_wait', 'congestion_metrics_cache', 'congestion_metrics_cache_timeout', 'congestion_control', 'deferred_accept', 'delay_window_control', 'delayed_acks', 'dsack', 'early_retransmit', 'explicit_congestion_notification', 'enhanced_loss_recovery', 'fast_open', 'fast_open_cookie_expiration', 'fin_wait_1', 'fin_wait_2', 'idle_timeout', 'initial_congestion_window_size', 'initial_receive_window_size', 'dont_fragment_flag', 'ip_tos', 'time_to_live', 'time_to_live_v4', 'time_to_live_v6', 'keep_alive_interval', 'limited_transmit_recovery', 'link_qos', 'max_segment_retrans', 'max_syn_retrans', 'max_segment_size', 'md5_signature', 'minimum_rto', 'multipath_tcp', 'mptcp_checksum', 'mptcp_checksum_verify', 'mptcp_fallback', 'mptcp_fast_join', 'mptcp_idle_timeout', 'mptcp_join_max', 'mptcp_make_after_break', 'mptcp_no_join_dss_ack', 'mptcp_rto_max', 'mptcp_retransmit_min', 'mptcp_subflow_max', 'mptcp_timeout', 'nagle_algorithm', 'pkt_loss_ignore_burst', 'pkt_loss_ignore_rate', 'proxy_buffer_high', 'proxy_buffer_low', 'proxy_max_segment', 'proxy_options', 'push_flag', 'rate_pace', 'rate_pace_max_rate', 'receive_window', 'reset_on_timeout', 'retransmit_threshold', 'selective_acks', 'selective_nack', 'send_buffer', 'slow_start', 'syn_cookie_enable', 'syn_cookie_white_list', 'syn_retrans_to_base', 'tail_loss_probe', 'time_wait_recycle', 'time_wait', 'timestamps', 'verified_accept', 'zero_window_timeout']

    @property
    def description(self):
        if self._values['description'] in [None, 'none']:
            return None
        return self._values['description']

    @property
    def time_wait(self):
        if self._values['time_wait'] is None:
            return None
        if self._values['time_wait'] == 0:
            return 'immediate'
        if self._values['time_wait'] == 4294967295:
            return 'indefinite'
        return self._values['time_wait']

    @property
    def close_wait(self):
        if self._values['close_wait'] is None:
            return None
        if self._values['close_wait'] == 0:
            return 'immediate'
        if self._values['close_wait'] == 4294967295:
            return 'indefinite'
        return self._values['close_wait']

    @property
    def fin_wait_1(self):
        if self._values['fin_wait_1'] is None:
            return None
        if self._values['fin_wait_1'] == 0:
            return 'immediate'
        if self._values['fin_wait_1'] == 4294967295:
            return 'indefinite'
        return self._values['fin_wait_1']

    @property
    def fin_wait_2(self):
        if self._values['fin_wait_2'] is None:
            return None
        if self._values['fin_wait_2'] == 0:
            return 'immediate'
        if self._values['fin_wait_2'] == 4294967295:
            return 'indefinite'
        return self._values['fin_wait_2']

    @property
    def zero_window_timeout(self):
        if self._values['zero_window_timeout'] is None:
            return None
        if self._values['zero_window_timeout'] == 4294967295:
            return 'indefinite'
        return self._values['zero_window_timeout']

    @property
    def idle_timeout(self):
        if self._values['idle_timeout'] is None:
            return None
        if self._values['idle_timeout'] == 4294967295:
            return 'indefinite'
        return self._values['idle_timeout']

    @property
    def keep_alive_interval(self):
        if self._values['keep_alive_interval'] is None:
            return None
        if self._values['keep_alive_interval'] == 4294967295:
            return 'indefinite'
        return self._values['keep_alive_interval']

    @property
    def verified_accept(self):
        return flatten_boolean(self._values['verified_accept'])

    @property
    def timestamps(self):
        return flatten_boolean(self._values['timestamps'])

    @property
    def time_wait_recycle(self):
        return flatten_boolean(self._values['time_wait_recycle'])

    @property
    def tail_loss_probe(self):
        return flatten_boolean(self._values['tail_loss_probe'])

    @property
    def syn_cookie_white_list(self):
        return flatten_boolean(self._values['syn_cookie_white_list'])

    @property
    def syn_cookie_enable(self):
        return flatten_boolean(self._values['syn_cookie_enable'])

    @property
    def slow_start(self):
        return flatten_boolean(self._values['slow_start'])

    @property
    def selective_nack(self):
        return flatten_boolean(self._values['selective_nack'])

    @property
    def selective_acks(self):
        return flatten_boolean(self._values['selective_acks'])

    @property
    def reset_on_timeout(self):
        return flatten_boolean(self._values['reset_on_timeout'])

    @property
    def rate_pace(self):
        return flatten_boolean(self._values['rate_pace'])

    @property
    def proxy_options(self):
        return flatten_boolean(self._values['proxy_options'])

    @property
    def proxy_max_segment(self):
        return flatten_boolean(self._values['proxy_max_segment'])

    @property
    def nagle_algorithm(self):
        return flatten_boolean(self._values['nagle_algorithm'])

    @property
    def mptcp_no_join_dss_ack(self):
        return flatten_boolean(self._values['mptcp_no_join_dss_ack'])

    @property
    def mptcp_make_after_break(self):
        return flatten_boolean(self._values['mptcp_make_after_break'])

    @property
    def mptcp_fast_join(self):
        return flatten_boolean(self._values['mptcp_fast_join'])

    @property
    def mptcp_checksum_verify(self):
        return flatten_boolean(self._values['mptcp_checksum_verify'])

    @property
    def mptcp_checksum(self):
        return flatten_boolean(self._values['mptcp_checksum'])

    @property
    def multipath_tcp(self):
        return flatten_boolean(self._values['multipath_tcp'])

    @property
    def md5_signature(self):
        return flatten_boolean(self._values['md5_signature'])

    @property
    def limited_transmit_recovery(self):
        return flatten_boolean(self._values['limited_transmit_recovery'])

    @property
    def fast_open(self):
        return flatten_boolean(self._values['fast_open'])

    @property
    def enhanced_loss_recovery(self):
        return flatten_boolean(self._values['enhanced_loss_recovery'])

    @property
    def explicit_congestion_notification(self):
        return flatten_boolean(self._values['explicit_congestion_notification'])

    @property
    def early_retransmit(self):
        return flatten_boolean(self._values['early_retransmit'])

    @property
    def dsack(self):
        return flatten_boolean(self._values['dsack'])

    @property
    def delayed_acks(self):
        return flatten_boolean(self._values['delayed_acks'])

    @property
    def delay_window_control(self):
        return flatten_boolean(self._values['delay_window_control'])

    @property
    def deferred_accept(self):
        return flatten_boolean(self._values['deferred_accept'])

    @property
    def congestion_metrics_cache(self):
        return flatten_boolean(self._values['congestion_metrics_cache'])

    @property
    def auto_send_buffer(self):
        return flatten_boolean(self._values['auto_send_buffer'])

    @property
    def auto_receive_window(self):
        return flatten_boolean(self._values['auto_receive_window'])

    @property
    def auto_proxy_buffer(self):
        return flatten_boolean(self._values['auto_proxy_buffer'])

    @property
    def abc(self):
        return flatten_boolean(self._values['abc'])

    @property
    def ack_on_push(self):
        return flatten_boolean(self._values['ack_on_push'])