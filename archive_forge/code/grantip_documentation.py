from paste.util import ip4

    Grant roles or usernames based on IP addresses.

    Config looks like this::

      [filter:grant]
      use = egg:Paste#grantip
      clobber_username = true
      # Give localhost system role (no username):
      127.0.0.1 = -:system
      # Give everyone in 192.168.0.* editor role:
      192.168.0.0/24 = -:editor
      # Give one IP the username joe:
      192.168.0.7 = joe
      # And one IP is should not be logged in:
      192.168.0.10 = __remove__:-editor

    