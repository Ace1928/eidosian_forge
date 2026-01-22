import os
class IObject(object):

    def choose_from_list(self, item_list, search_str='', prompt='Enter Selection'):
        if not item_list:
            print('No Choices Available')
            return
        choice = None
        while not choice:
            n = 1
            choices = []
            for item in item_list:
                if isinstance(item, basestring):
                    print('[%d] %s' % (n, item))
                    choices.append(item)
                    n += 1
                else:
                    obj, id, desc = item
                    if desc:
                        if desc.find(search_str) >= 0:
                            print('[%d] %s - %s' % (n, id, desc))
                            choices.append(obj)
                            n += 1
                    elif id.find(search_str) >= 0:
                        print('[%d] %s' % (n, id))
                        choices.append(obj)
                        n += 1
            if choices:
                val = raw_input('%s[1-%d]: ' % (prompt, len(choices)))
                if val.startswith('/'):
                    search_str = val[1:]
                else:
                    try:
                        int_val = int(val)
                        if int_val == 0:
                            return None
                        choice = choices[int_val - 1]
                    except ValueError:
                        print('%s is not a valid choice' % val)
                    except IndexError:
                        print('%s is not within the range[1-%d]' % (val, len(choices)))
            else:
                print('No objects matched your pattern')
                search_str = ''
        return choice

    def get_string(self, prompt, validation_fn=None):
        okay = False
        while not okay:
            val = raw_input('%s: ' % prompt)
            if validation_fn:
                okay = validation_fn(val)
                if not okay:
                    print('Invalid value: %s' % val)
            else:
                okay = True
        return val

    def get_filename(self, prompt):
        okay = False
        val = ''
        while not okay:
            val = raw_input('%s: %s' % (prompt, val))
            val = os.path.expanduser(val)
            if os.path.isfile(val):
                okay = True
            elif os.path.isdir(val):
                path = val
                val = self.choose_from_list(os.listdir(path))
                if val:
                    val = os.path.join(path, val)
                    okay = True
                else:
                    val = ''
            else:
                print('Invalid value: %s' % val)
                val = ''
        return val

    def get_int(self, prompt):
        s = self.get_string(prompt, int_val_fn)
        return int(s)