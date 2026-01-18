from __future__ import absolute_import, division, print_function
import os.path
import re
from ansible_collections.community.dns.plugins.module_utils.names import InvalidDomainName, split_into_labels, normalize_label

        Given a domain name, extracts the registrable domain. This is the public suffix
        including the last label before the suffix.

        If ``keep_unknown_suffix`` is set to ``False``, only suffixes matching explicit
        entries from the PSL are returned. If no suffix can be found, ``''`` is returned.
        If ``keep_unknown_suffix`` is ``True`` (default), the implicit ``*`` rule is used
        if no other rule matches.

        If ``only_if_registerable`` is set to ``False``, the public suffix is returned
        if there is no label before the suffix. If ``only_if_registerable`` is ``True``
        (default), ``''`` is returned in that case.

        If ``normalize_result`` is set to ``True``, the result is re-combined form the
        normalized labels. In that case, the result is lower-case ASCII. If
        ``normalize_result`` is ``False`` (default), the result ``result`` always satisfies
        ``domain.endswith(result)``.

        If ``icann_only`` is set to ``True``, only official ICANN rules are used. If
        ``icann_only`` is ``False`` (default), also private rules are used.
        