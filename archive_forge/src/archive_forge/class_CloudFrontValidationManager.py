import datetime
import re
from collections import OrderedDict
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class CloudFrontValidationManager(object):
    """
    Manages CloudFront validations
    """

    def __init__(self, module):
        self.__cloudfront_facts_mgr = CloudFrontFactsServiceManager(module)
        self.module = module
        self.__default_distribution_enabled = True
        self.__default_http_port = 80
        self.__default_https_port = 443
        self.__default_ipv6_enabled = False
        self.__default_origin_ssl_protocols = ['TLSv1', 'TLSv1.1', 'TLSv1.2']
        self.__default_custom_origin_protocol_policy = 'match-viewer'
        self.__default_custom_origin_read_timeout = 30
        self.__default_custom_origin_keepalive_timeout = 5
        self.__default_datetime_string = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        self.__default_cache_behavior_min_ttl = 0
        self.__default_cache_behavior_max_ttl = 31536000
        self.__default_cache_behavior_default_ttl = 86400
        self.__default_cache_behavior_compress = False
        self.__default_cache_behavior_viewer_protocol_policy = 'allow-all'
        self.__default_cache_behavior_smooth_streaming = False
        self.__default_cache_behavior_forwarded_values_forward_cookies = 'none'
        self.__default_cache_behavior_forwarded_values_query_string = True
        self.__default_trusted_signers_enabled = False
        self.__valid_price_classes = set(['PriceClass_100', 'PriceClass_200', 'PriceClass_All'])
        self.__valid_origin_protocol_policies = set(['http-only', 'match-viewer', 'https-only'])
        self.__valid_origin_ssl_protocols = set(['SSLv3', 'TLSv1', 'TLSv1.1', 'TLSv1.2'])
        self.__valid_cookie_forwarding = set(['none', 'whitelist', 'all'])
        self.__valid_viewer_protocol_policies = set(['allow-all', 'https-only', 'redirect-to-https'])
        self.__valid_methods = set(['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'OPTIONS', 'DELETE'])
        self.__valid_methods_cached_methods = [set(['GET', 'HEAD']), set(['GET', 'HEAD', 'OPTIONS'])]
        self.__valid_methods_allowed_methods = [self.__valid_methods_cached_methods[0], self.__valid_methods_cached_methods[1], self.__valid_methods]
        self.__valid_lambda_function_association_event_types = set(['viewer-request', 'viewer-response', 'origin-request', 'origin-response'])
        self.__valid_viewer_certificate_ssl_support_methods = set(['sni-only', 'vip'])
        self.__valid_viewer_certificate_minimum_protocol_versions = set(['SSLv3', 'TLSv1', 'TLSv1_2016', 'TLSv1.1_2016', 'TLSv1.2_2018', 'TLSv1.2_2019', 'TLSv1.2_2021'])
        self.__valid_viewer_certificate_certificate_sources = set(['cloudfront', 'iam', 'acm'])
        self.__valid_http_versions = set(['http1.1', 'http2', 'http3', 'http2and3'])
        self.__s3_bucket_domain_regex = re.compile('\\.s3(?:\\.[^.]+)?\\.amazonaws\\.com$')

    def add_missing_key(self, dict_object, key_to_set, value_to_set):
        if key_to_set not in dict_object and value_to_set is not None:
            dict_object[key_to_set] = value_to_set
        return dict_object

    def add_key_else_change_dict_key(self, dict_object, old_key, new_key, value_to_set):
        if old_key not in dict_object and value_to_set is not None:
            dict_object[new_key] = value_to_set
        else:
            dict_object = change_dict_key_name(dict_object, old_key, new_key)
        return dict_object

    def add_key_else_validate(self, dict_object, key_name, attribute_name, value_to_set, valid_values, to_aws_list=False):
        if key_name in dict_object:
            self.validate_attribute_with_allowed_values(value_to_set, attribute_name, valid_values)
        elif to_aws_list:
            dict_object[key_name] = ansible_list_to_cloudfront_list(value_to_set)
        elif value_to_set is not None:
            dict_object[key_name] = value_to_set
        return dict_object

    def validate_logging(self, logging):
        try:
            if logging is None:
                return None
            valid_logging = {}
            if logging and (not set(['enabled', 'include_cookies', 'bucket', 'prefix']).issubset(logging)):
                self.module.fail_json(msg='The logging parameters enabled, include_cookies, bucket and prefix must be specified.')
            valid_logging['include_cookies'] = logging.get('include_cookies')
            valid_logging['enabled'] = logging.get('enabled')
            valid_logging['bucket'] = logging.get('bucket')
            valid_logging['prefix'] = logging.get('prefix')
            return valid_logging
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating distribution logging')

    def validate_is_list(self, list_to_validate, list_name):
        if not isinstance(list_to_validate, list):
            self.module.fail_json(msg=f'{list_name} is of type {type(list_to_validate).__name__}. Must be a list.')

    def validate_required_key(self, key_name, full_key_name, dict_object):
        if key_name not in dict_object:
            self.module.fail_json(msg=f'{full_key_name} must be specified.')

    def validate_origins(self, client, config, origins, default_origin_domain_name, default_origin_path, create_distribution, purge_origins=False):
        try:
            if origins is None:
                if default_origin_domain_name is None and (not create_distribution):
                    if purge_origins:
                        return None
                    else:
                        return ansible_list_to_cloudfront_list(config)
                if default_origin_domain_name is not None:
                    origins = [{'domain_name': default_origin_domain_name, 'origin_path': default_origin_path or ''}]
                else:
                    origins = []
            self.validate_is_list(origins, 'origins')
            if not origins and default_origin_domain_name is None and create_distribution:
                self.module.fail_json(msg='Both origins[] and default_origin_domain_name have not been specified. Please specify at least one.')
            all_origins = OrderedDict()
            new_domains = list()
            for origin in config:
                all_origins[origin.get('domain_name')] = origin
            for origin in origins:
                origin = self.validate_origin(client, all_origins.get(origin.get('domain_name'), {}), origin, default_origin_path)
                all_origins[origin['domain_name']] = origin
                new_domains.append(origin['domain_name'])
            if purge_origins:
                for domain in list(all_origins.keys()):
                    if domain not in new_domains:
                        del all_origins[domain]
            return ansible_list_to_cloudfront_list(list(all_origins.values()))
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating distribution origins')

    def validate_s3_origin_configuration(self, client, existing_config, origin):
        if origin.get('s3_origin_config', {}).get('origin_access_identity'):
            return origin['s3_origin_config']['origin_access_identity']
        if existing_config.get('s3_origin_config', {}).get('origin_access_identity'):
            return existing_config['s3_origin_config']['origin_access_identity']
        try:
            comment = f'access-identity-by-ansible-{origin.get('domain_name')}-{self.__default_datetime_string}'
            caller_reference = f'{origin.get('domain_name')}-{self.__default_datetime_string}'
            cfoai_config = dict(CloudFrontOriginAccessIdentityConfig=dict(CallerReference=caller_reference, Comment=comment))
            oai = client.create_cloud_front_origin_access_identity(**cfoai_config)['CloudFrontOriginAccessIdentity']['Id']
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f"Couldn't create Origin Access Identity for id {origin['id']}")
        return f'origin-access-identity/cloudfront/{oai}'

    def validate_origin(self, client, existing_config, origin, default_origin_path):
        try:
            origin = self.add_missing_key(origin, 'origin_path', existing_config.get('origin_path', default_origin_path or ''))
            self.validate_required_key('origin_path', 'origins[].origin_path', origin)
            origin = self.add_missing_key(origin, 'id', existing_config.get('id', self.__default_datetime_string))
            if 'custom_headers' in origin and len(origin.get('custom_headers')) > 0:
                for custom_header in origin.get('custom_headers'):
                    if 'header_name' not in custom_header or 'header_value' not in custom_header:
                        self.module.fail_json(msg='Both origins[].custom_headers.header_name and origins[].custom_headers.header_value must be specified.')
                origin['custom_headers'] = ansible_list_to_cloudfront_list(origin.get('custom_headers'))
            else:
                origin['custom_headers'] = ansible_list_to_cloudfront_list()
            if 'origin_shield' in origin:
                origin_shield = origin.get('origin_shield')
                if origin_shield.get('enabled'):
                    origin_shield_region = origin_shield.get('origin_shield_region')
                    if origin_shield_region is None:
                        self.module.fail_json(msg='origins[].origin_shield.origin_shield_region must be specified when origins[].origin_shield.enabled is true.')
                    else:
                        origin_shield_region = origin_shield_region.lower()
            if self.__s3_bucket_domain_regex.search(origin.get('domain_name').lower()):
                if origin.get('s3_origin_access_identity_enabled') is not None:
                    if origin['s3_origin_access_identity_enabled']:
                        s3_origin_config = self.validate_s3_origin_configuration(client, existing_config, origin)
                    else:
                        s3_origin_config = None
                    del origin['s3_origin_access_identity_enabled']
                    if s3_origin_config:
                        oai = s3_origin_config
                    else:
                        oai = ''
                    origin['s3_origin_config'] = dict(origin_access_identity=oai)
                if 'custom_origin_config' in origin:
                    self.module.fail_json(msg='s3 origin domains and custom_origin_config are mutually exclusive')
            else:
                origin = self.add_missing_key(origin, 'custom_origin_config', existing_config.get('custom_origin_config', {}))
                custom_origin_config = origin.get('custom_origin_config')
                custom_origin_config = self.add_key_else_validate(custom_origin_config, 'origin_protocol_policy', 'origins[].custom_origin_config.origin_protocol_policy', self.__default_custom_origin_protocol_policy, self.__valid_origin_protocol_policies)
                custom_origin_config = self.add_missing_key(custom_origin_config, 'origin_read_timeout', self.__default_custom_origin_read_timeout)
                custom_origin_config = self.add_missing_key(custom_origin_config, 'origin_keepalive_timeout', self.__default_custom_origin_keepalive_timeout)
                custom_origin_config = self.add_key_else_change_dict_key(custom_origin_config, 'http_port', 'h_t_t_p_port', self.__default_http_port)
                custom_origin_config = self.add_key_else_change_dict_key(custom_origin_config, 'https_port', 'h_t_t_p_s_port', self.__default_https_port)
                if custom_origin_config.get('origin_ssl_protocols', {}).get('items'):
                    custom_origin_config['origin_ssl_protocols'] = custom_origin_config['origin_ssl_protocols']['items']
                if custom_origin_config.get('origin_ssl_protocols'):
                    self.validate_attribute_list_with_allowed_list(custom_origin_config['origin_ssl_protocols'], 'origins[].origin_ssl_protocols', self.__valid_origin_ssl_protocols)
                else:
                    custom_origin_config['origin_ssl_protocols'] = self.__default_origin_ssl_protocols
                custom_origin_config['origin_ssl_protocols'] = ansible_list_to_cloudfront_list(custom_origin_config['origin_ssl_protocols'])
            return origin
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Error validating distribution origin')

    def validate_cache_behaviors(self, config, cache_behaviors, valid_origins, purge_cache_behaviors=False):
        try:
            if cache_behaviors is None and valid_origins is not None and (purge_cache_behaviors is False):
                return ansible_list_to_cloudfront_list(config)
            all_cache_behaviors = OrderedDict()
            if not purge_cache_behaviors:
                for behavior in config:
                    all_cache_behaviors[behavior['path_pattern']] = behavior
            for cache_behavior in cache_behaviors:
                valid_cache_behavior = self.validate_cache_behavior(all_cache_behaviors.get(cache_behavior.get('path_pattern'), {}), cache_behavior, valid_origins)
                all_cache_behaviors[cache_behavior['path_pattern']] = valid_cache_behavior
            if purge_cache_behaviors:
                for target_origin_id in set(all_cache_behaviors.keys()) - set([cb['path_pattern'] for cb in cache_behaviors]):
                    del all_cache_behaviors[target_origin_id]
            return ansible_list_to_cloudfront_list(list(all_cache_behaviors.values()))
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating distribution cache behaviors')

    def validate_cache_behavior(self, config, cache_behavior, valid_origins, is_default_cache=False):
        if is_default_cache and cache_behavior is None:
            cache_behavior = {}
        if cache_behavior is None and valid_origins is not None:
            return config
        cache_behavior = self.validate_cache_behavior_first_level_keys(config, cache_behavior, valid_origins, is_default_cache)
        if cache_behavior.get('cache_policy_id') is None:
            cache_behavior = self.validate_forwarded_values(config, cache_behavior.get('forwarded_values'), cache_behavior)
        cache_behavior = self.validate_allowed_methods(config, cache_behavior.get('allowed_methods'), cache_behavior)
        cache_behavior = self.validate_lambda_function_associations(config, cache_behavior.get('lambda_function_associations'), cache_behavior)
        cache_behavior = self.validate_trusted_signers(config, cache_behavior.get('trusted_signers'), cache_behavior)
        cache_behavior = self.validate_field_level_encryption_id(config, cache_behavior.get('field_level_encryption_id'), cache_behavior)
        return cache_behavior

    def validate_cache_behavior_first_level_keys(self, config, cache_behavior, valid_origins, is_default_cache):
        if cache_behavior.get('cache_policy_id') is not None and cache_behavior.get('forwarded_values') is not None:
            if is_default_cache:
                cache_behavior_name = 'Default cache behavior'
            else:
                cache_behavior_name = f'Cache behavior for path {cache_behavior['path_pattern']}'
            self.module.fail_json(msg=f'{cache_behavior_name} cannot have both a cache_policy_id and a forwarded_values option.')
        try:
            if cache_behavior.get('cache_policy_id') is None:
                cache_behavior = self.add_key_else_change_dict_key(cache_behavior, 'min_ttl', 'min_t_t_l', config.get('min_t_t_l', self.__default_cache_behavior_min_ttl))
                cache_behavior = self.add_key_else_change_dict_key(cache_behavior, 'max_ttl', 'max_t_t_l', config.get('max_t_t_l', self.__default_cache_behavior_max_ttl))
                cache_behavior = self.add_key_else_change_dict_key(cache_behavior, 'default_ttl', 'default_t_t_l', config.get('default_t_t_l', self.__default_cache_behavior_default_ttl))
            cache_behavior = self.add_missing_key(cache_behavior, 'compress', config.get('compress', self.__default_cache_behavior_compress))
            target_origin_id = cache_behavior.get('target_origin_id', config.get('target_origin_id'))
            if not target_origin_id:
                target_origin_id = self.get_first_origin_id_for_default_cache_behavior(valid_origins)
            if target_origin_id not in [origin['id'] for origin in valid_origins.get('items', [])]:
                if is_default_cache:
                    cache_behavior_name = 'Default cache behavior'
                else:
                    cache_behavior_name = f'Cache behavior for path {cache_behavior['path_pattern']}'
                self.module.fail_json(msg=f'{cache_behavior_name} has target_origin_id pointing to an origin that does not exist.')
            cache_behavior['target_origin_id'] = target_origin_id
            cache_behavior = self.add_key_else_validate(cache_behavior, 'viewer_protocol_policy', 'cache_behavior.viewer_protocol_policy', config.get('viewer_protocol_policy', self.__default_cache_behavior_viewer_protocol_policy), self.__valid_viewer_protocol_policies)
            cache_behavior = self.add_missing_key(cache_behavior, 'smooth_streaming', config.get('smooth_streaming', self.__default_cache_behavior_smooth_streaming))
            return cache_behavior
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating distribution cache behavior first level keys')

    def validate_forwarded_values(self, config, forwarded_values, cache_behavior):
        try:
            if not forwarded_values:
                forwarded_values = dict()
            existing_config = config.get('forwarded_values', {})
            headers = forwarded_values.get('headers', existing_config.get('headers', {}).get('items'))
            if headers:
                headers.sort()
            forwarded_values['headers'] = ansible_list_to_cloudfront_list(headers)
            if 'cookies' not in forwarded_values:
                forward = existing_config.get('cookies', {}).get('forward', self.__default_cache_behavior_forwarded_values_forward_cookies)
                forwarded_values['cookies'] = {'forward': forward}
            else:
                existing_whitelist = existing_config.get('cookies', {}).get('whitelisted_names', {}).get('items')
                whitelist = forwarded_values.get('cookies').get('whitelisted_names', existing_whitelist)
                if whitelist:
                    self.validate_is_list(whitelist, 'forwarded_values.whitelisted_names')
                    forwarded_values['cookies']['whitelisted_names'] = ansible_list_to_cloudfront_list(whitelist)
                cookie_forwarding = forwarded_values.get('cookies').get('forward', existing_config.get('cookies', {}).get('forward'))
                self.validate_attribute_with_allowed_values(cookie_forwarding, 'cache_behavior.forwarded_values.cookies.forward', self.__valid_cookie_forwarding)
                forwarded_values['cookies']['forward'] = cookie_forwarding
            query_string_cache_keys = forwarded_values.get('query_string_cache_keys', existing_config.get('query_string_cache_keys', {}).get('items', []))
            self.validate_is_list(query_string_cache_keys, 'forwarded_values.query_string_cache_keys')
            forwarded_values['query_string_cache_keys'] = ansible_list_to_cloudfront_list(query_string_cache_keys)
            forwarded_values = self.add_missing_key(forwarded_values, 'query_string', existing_config.get('query_string', self.__default_cache_behavior_forwarded_values_query_string))
            cache_behavior['forwarded_values'] = forwarded_values
            return cache_behavior
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating forwarded values')

    def validate_lambda_function_associations(self, config, lambda_function_associations, cache_behavior):
        try:
            if lambda_function_associations is not None:
                self.validate_is_list(lambda_function_associations, 'lambda_function_associations')
                for association in lambda_function_associations:
                    association = change_dict_key_name(association, 'lambda_function_arn', 'lambda_function_a_r_n')
                    self.validate_attribute_with_allowed_values(association.get('event_type'), 'cache_behaviors[].lambda_function_associations.event_type', self.__valid_lambda_function_association_event_types)
                cache_behavior['lambda_function_associations'] = ansible_list_to_cloudfront_list(lambda_function_associations)
            elif 'lambda_function_associations' in config:
                cache_behavior['lambda_function_associations'] = config.get('lambda_function_associations')
            else:
                cache_behavior['lambda_function_associations'] = ansible_list_to_cloudfront_list([])
            return cache_behavior
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating lambda function associations')

    def validate_field_level_encryption_id(self, config, field_level_encryption_id, cache_behavior):
        if field_level_encryption_id is not None:
            cache_behavior['field_level_encryption_id'] = field_level_encryption_id
        elif 'field_level_encryption_id' in config:
            cache_behavior['field_level_encryption_id'] = config.get('field_level_encryption_id')
        else:
            cache_behavior['field_level_encryption_id'] = ''
        return cache_behavior

    def validate_allowed_methods(self, config, allowed_methods, cache_behavior):
        try:
            if allowed_methods is not None:
                self.validate_required_key('items', 'cache_behavior.allowed_methods.items[]', allowed_methods)
                temp_allowed_items = allowed_methods.get('items')
                self.validate_is_list(temp_allowed_items, 'cache_behavior.allowed_methods.items')
                self.validate_attribute_list_with_allowed_list(temp_allowed_items, 'cache_behavior.allowed_methods.items[]', self.__valid_methods_allowed_methods)
                cached_items = allowed_methods.get('cached_methods')
                if 'cached_methods' in allowed_methods:
                    self.validate_is_list(cached_items, 'cache_behavior.allowed_methods.cached_methods')
                    self.validate_attribute_list_with_allowed_list(cached_items, 'cache_behavior.allowed_items.cached_methods[]', self.__valid_methods_cached_methods)
                if 'allowed_methods' in config and set(config['allowed_methods']['items']) == set(temp_allowed_items):
                    cache_behavior['allowed_methods'] = config['allowed_methods']
                else:
                    cache_behavior['allowed_methods'] = ansible_list_to_cloudfront_list(temp_allowed_items)
                if cached_items and set(cached_items) == set(config.get('allowed_methods', {}).get('cached_methods', {}).get('items', [])):
                    cache_behavior['allowed_methods']['cached_methods'] = config['allowed_methods']['cached_methods']
                else:
                    cache_behavior['allowed_methods']['cached_methods'] = ansible_list_to_cloudfront_list(cached_items)
            elif 'allowed_methods' in config:
                cache_behavior['allowed_methods'] = config.get('allowed_methods')
            return cache_behavior
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating allowed methods')

    def validate_trusted_signers(self, config, trusted_signers, cache_behavior):
        try:
            if trusted_signers is None:
                trusted_signers = {}
            if 'items' in trusted_signers:
                valid_trusted_signers = ansible_list_to_cloudfront_list(trusted_signers.get('items'))
            else:
                valid_trusted_signers = dict(quantity=config.get('quantity', 0))
                if 'items' in config:
                    valid_trusted_signers = dict(items=config['items'])
            valid_trusted_signers['enabled'] = trusted_signers.get('enabled', config.get('enabled', self.__default_trusted_signers_enabled))
            cache_behavior['trusted_signers'] = valid_trusted_signers
            return cache_behavior
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating trusted signers')

    def validate_viewer_certificate(self, viewer_certificate):
        try:
            if viewer_certificate is None:
                return None
            if viewer_certificate.get('cloudfront_default_certificate') and viewer_certificate.get('ssl_support_method') is not None:
                self.module.fail_json(msg='viewer_certificate.ssl_support_method should not be specified with viewer_certificate_cloudfront_default' + '_certificate set to true.')
            self.validate_attribute_with_allowed_values(viewer_certificate.get('ssl_support_method'), 'viewer_certificate.ssl_support_method', self.__valid_viewer_certificate_ssl_support_methods)
            self.validate_attribute_with_allowed_values(viewer_certificate.get('minimum_protocol_version'), 'viewer_certificate.minimum_protocol_version', self.__valid_viewer_certificate_minimum_protocol_versions)
            self.validate_attribute_with_allowed_values(viewer_certificate.get('certificate_source'), 'viewer_certificate.certificate_source', self.__valid_viewer_certificate_certificate_sources)
            viewer_certificate = change_dict_key_name(viewer_certificate, 'cloudfront_default_certificate', 'cloud_front_default_certificate')
            viewer_certificate = change_dict_key_name(viewer_certificate, 'ssl_support_method', 's_s_l_support_method')
            viewer_certificate = change_dict_key_name(viewer_certificate, 'iam_certificate_id', 'i_a_m_certificate_id')
            viewer_certificate = change_dict_key_name(viewer_certificate, 'acm_certificate_arn', 'a_c_m_certificate_arn')
            return viewer_certificate
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating viewer certificate')

    def validate_custom_error_responses(self, config, custom_error_responses, purge_custom_error_responses):
        try:
            if custom_error_responses is None and (not purge_custom_error_responses):
                return ansible_list_to_cloudfront_list(config)
            self.validate_is_list(custom_error_responses, 'custom_error_responses')
            result = list()
            existing_responses = dict(((response['error_code'], response) for response in custom_error_responses))
            for custom_error_response in custom_error_responses:
                self.validate_required_key('error_code', 'custom_error_responses[].error_code', custom_error_response)
                custom_error_response = change_dict_key_name(custom_error_response, 'error_caching_min_ttl', 'error_caching_min_t_t_l')
                if 'response_code' in custom_error_response:
                    custom_error_response['response_code'] = str(custom_error_response['response_code'])
                if custom_error_response['error_code'] in existing_responses:
                    del existing_responses[custom_error_response['error_code']]
                result.append(custom_error_response)
            if not purge_custom_error_responses:
                result.extend(existing_responses.values())
            return ansible_list_to_cloudfront_list(result)
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating custom error responses')

    def validate_restrictions(self, config, restrictions, purge_restrictions=False):
        try:
            if restrictions is None:
                if purge_restrictions:
                    return None
                else:
                    return config
            self.validate_required_key('geo_restriction', 'restrictions.geo_restriction', restrictions)
            geo_restriction = restrictions.get('geo_restriction')
            self.validate_required_key('restriction_type', 'restrictions.geo_restriction.restriction_type', geo_restriction)
            existing_restrictions = config.get('geo_restriction', {}).get(geo_restriction['restriction_type'], {}).get('items', [])
            geo_restriction_items = geo_restriction.get('items')
            if not purge_restrictions:
                geo_restriction_items.extend([rest for rest in existing_restrictions if rest not in geo_restriction_items])
            valid_restrictions = ansible_list_to_cloudfront_list(geo_restriction_items)
            valid_restrictions['restriction_type'] = geo_restriction.get('restriction_type')
            return {'geo_restriction': valid_restrictions}
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating restrictions')

    def validate_distribution_config_parameters(self, config, default_root_object, ipv6_enabled, http_version, web_acl_id):
        try:
            config['default_root_object'] = default_root_object or config.get('default_root_object', '')
            config['is_i_p_v6_enabled'] = ipv6_enabled if ipv6_enabled is not None else config.get('is_i_p_v6_enabled', self.__default_ipv6_enabled)
            if http_version is not None or config.get('http_version'):
                self.validate_attribute_with_allowed_values(http_version, 'http_version', self.__valid_http_versions)
                config['http_version'] = http_version or config.get('http_version')
            if web_acl_id or config.get('web_a_c_l_id'):
                config['web_a_c_l_id'] = web_acl_id or config.get('web_a_c_l_id')
            return config
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating distribution config parameters')

    def validate_common_distribution_parameters(self, config, enabled, aliases, logging, price_class, purge_aliases=False):
        try:
            if config is None:
                config = {}
            if aliases is not None:
                if not purge_aliases:
                    aliases.extend([alias for alias in config.get('aliases', {}).get('items', []) if alias not in aliases])
                config['aliases'] = ansible_list_to_cloudfront_list(aliases)
            if logging is not None:
                config['logging'] = self.validate_logging(logging)
            config['enabled'] = enabled if enabled is not None else config.get('enabled', self.__default_distribution_enabled)
            if price_class is not None:
                self.validate_attribute_with_allowed_values(price_class, 'price_class', self.__valid_price_classes)
                config['price_class'] = price_class
            return config
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating common distribution parameters')

    def validate_comment(self, config, comment):
        config['comment'] = comment or config.get('comment', 'Distribution created by Ansible with datetime stamp ' + self.__default_datetime_string)
        return config

    def validate_caller_reference(self, caller_reference):
        return caller_reference or self.__default_datetime_string

    def get_first_origin_id_for_default_cache_behavior(self, valid_origins):
        try:
            if valid_origins is not None:
                valid_origins_list = valid_origins.get('items')
                if valid_origins_list is not None and isinstance(valid_origins_list, list) and (len(valid_origins_list) > 0):
                    return str(valid_origins_list[0].get('id'))
            self.module.fail_json(msg='There are no valid origins from which to specify a target_origin_id for the default_cache_behavior configuration.')
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error getting first origin_id for default cache behavior')

    def validate_attribute_list_with_allowed_list(self, attribute_list, attribute_list_name, allowed_list):
        try:
            self.validate_is_list(attribute_list, attribute_list_name)
            if isinstance(allowed_list, list) and set(attribute_list) not in allowed_list or (isinstance(allowed_list, set) and (not set(allowed_list).issuperset(attribute_list))):
                attribute_list = ' '.join((str(a) for a in allowed_list))
                self.module.fail_json(msg=f'The attribute list {attribute_list_name} must be one of [{attribute_list}]')
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating attribute list with allowed value list')

    def validate_attribute_with_allowed_values(self, attribute, attribute_name, allowed_list):
        if attribute is not None and attribute not in allowed_list:
            attribute_list = ' '.join((str(a) for a in allowed_list))
            self.module.fail_json(msg=f'The attribute {attribute_name} must be one of [{attribute_list}]')

    def validate_distribution_from_caller_reference(self, caller_reference):
        try:
            distributions = self.__cloudfront_facts_mgr.list_distributions(keyed=False)
            distribution_name = 'Distribution'
            distribution_config_name = 'DistributionConfig'
            distribution_ids = [dist.get('Id') for dist in distributions]
            for distribution_id in distribution_ids:
                distribution = self.__cloudfront_facts_mgr.get_distribution(id=distribution_id)
                if distribution is not None:
                    distribution_config = distribution[distribution_name].get(distribution_config_name)
                    if distribution_config is not None and distribution_config.get('CallerReference') == caller_reference:
                        distribution[distribution_name][distribution_config_name] = distribution_config
                        return distribution
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating distribution from caller reference')

    def validate_distribution_from_aliases_caller_reference(self, distribution_id, aliases, caller_reference):
        try:
            if caller_reference is not None:
                return self.validate_distribution_from_caller_reference(caller_reference)
            else:
                if aliases and distribution_id is None:
                    distribution_id = self.validate_distribution_id_from_alias(aliases)
                if distribution_id:
                    return self.__cloudfront_facts_mgr.get_distribution(id=distribution_id)
            return None
        except Exception as e:
            self.module.fail_json_aws(e, msg='Error validating distribution_id from alias, aliases and caller reference')

    def validate_distribution_id_from_alias(self, aliases):
        distributions = self.__cloudfront_facts_mgr.list_distributions(keyed=False)
        if distributions:
            for distribution in distributions:
                distribution_aliases = distribution.get('Aliases', {}).get('Items', [])
                if set(aliases) & set(distribution_aliases):
                    return distribution['Id']
        return None

    def wait_until_processed(self, client, wait_timeout, distribution_id, caller_reference):
        if distribution_id is None:
            distribution = self.validate_distribution_from_caller_reference(caller_reference=caller_reference)
            distribution_id = distribution['Distribution']['Id']
        try:
            waiter = client.get_waiter('distribution_deployed')
            attempts = 1 + int(wait_timeout / 60)
            waiter.wait(Id=distribution_id, WaiterConfig={'MaxAttempts': attempts})
        except botocore.exceptions.WaiterError as e:
            self.module.fail_json_aws(e, msg=f'Timeout waiting for CloudFront action. Waited for {to_text(wait_timeout)} seconds before timeout.')
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg=f'Error getting distribution {distribution_id}')